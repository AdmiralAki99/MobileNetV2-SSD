# ---- Find the Deep Learning AMI ----
# "data" blocks READ information from AWS (don't create anything)
# This finds the latest Deep Learning AMI with NVIDIA drivers pre-installed
data "aws_ami" "deep_learning" {
  most_recent = true
  owners      = ["amazon"]

  filter {
    name   = "name"
    values = ["Deep Learning OSS Nvidia Driver AMI GPU TensorFlow 2.17 (Ubuntu 22.04)*"]
  }

  filter {
    name   = "architecture"
    values = ["x86_64"]
  }

  filter {
    name   = "state"
    values = ["available"]
  }

  filter {
    name   = "root-device-type"
    values = ["ebs"]
  }
}

# ---- Discover all subnets in the default VPC ----

data "aws_vpc" "default" {
  default = true
}

data "aws_subnets" "default" {
  filter {
    name   = "vpc-id"
    values = [data.aws_vpc.default.id]
  }
}

# ---- Launch Template ----

resource "aws_launch_template" "training" {
  name_prefix = "ml-training-"
  image_id    = data.aws_ami.deep_learning.id

  tag_specifications {
    resource_type = "instance"
    tags = {
      Name = "ml-training"
      ManagedBy = "terraform"
    }
  }

  iam_instance_profile {
    name = aws_iam_instance_profile.training.name
  }

  network_interfaces {
    associate_public_ip_address = true
    security_groups             = [aws_security_group.training.id]
  }

  key_name = var.key_pair_name

  # Storage: 100GB root volume (for Docker images + dataset cache)
  block_device_mappings {
    device_name = "/dev/sda1"
    ebs {
      volume_size = 100
      volume_type = "gp3"
    }
  }

  # Bootstrap script that runs on first boot
  user_data = base64encode(templatefile("${path.module}/user_data.sh", {
    docker_image         = var.docker_image
    experiment_config    = var.experiment_config
    experiment_bucket    = var.experiment_bucket
    git_commit           = var.git_commit
    s3_checkpoint_bucket = var.s3_checkpoint_bucket
    s3_dataset_bucket    = var.s3_dataset_bucket
    use_tfrecords        = var.use_tfrecords
    dataset_name         = var.dataset_name
    dynamodb_table_name  = var.dynamodb_table_name
    aws_region           = var.aws_region
  }))

  tags = {
    Name      = "ml-training-${var.instance_type}"
    ManagedBy = "terraform"
  }
}

# ---- EC2 Fleet (maintain) ----

resource "aws_ec2_fleet" "training" {
  type                        = "maintain"
  terminate_instances         = true  # Terminate instances on terraform destroy
  terminate_instances_with_expiration = true

  launch_template_config {
    launch_template_specification {
      launch_template_id = aws_launch_template.training.id
      version            = "$Latest"
    }

    dynamic "override" {
      for_each = {
        for pair in setproduct(tolist(data.aws_subnets.default.ids), var.instance_types) :
        "${pair[0]}-${pair[1]}" => {
          subnet_id     = pair[0]
          instance_type = pair[1]
        }
      }
      content {
        subnet_id     = override.value.subnet_id
        instance_type = override.value.instance_type
      }
    }
  }

  spot_options {
    allocation_strategy = "price-capacity-optimized"
  }

  target_capacity_specification {
    default_target_capacity_type = "on-demand"
    total_target_capacity        = 1
  }

  tags = {
    Name      = "ml-training-${var.instance_type}"
    ManagedBy = "terraform"
  }
}

# ---- Wait for the instance to reach running state ----

resource "null_resource" "wait_for_instance" {
  depends_on = [aws_ec2_fleet.training]

  provisioner "local-exec" {
    # sed strips Windows CRLF line endings before bash executes the script
    command = "sed 's/\\r//' ${path.module}/wait_for_instance.sh | bash -s ${aws_ec2_fleet.training.id} ${var.aws_region}"
  }
}

# ---- Look up the instance the fleet created ----

data "aws_instances" "training" {
  depends_on = [null_resource.wait_for_instance]

  filter {
    name   = "tag:aws:ec2:fleet-id"
    values = [aws_ec2_fleet.training.id]
  }
}
