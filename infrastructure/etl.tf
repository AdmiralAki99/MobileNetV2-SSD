data "aws_ami" "deep_learning_pytorch" {
  most_recent = true
  owners      = ["amazon"]

  filter {
    name   = "name"
    values = ["Deep Learning OSS Nvidia Driver AMI GPU PyTorch *(Ubuntu 22.04)*"]
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

# ---- Launch Template ----

resource "aws_launch_template" "etl" {
  name_prefix = "ml-etl-"
  image_id    = data.aws_ami.deep_learning_pytorch.id
  instance_type = var.instance_type

  iam_instance_profile {
    name = aws_iam_instance_profile.training.name
  }

  network_interfaces {
    associate_public_ip_address = true
    security_groups = [aws_security_group.training.id]
  }

  key_name = var.key_pair_name

  block_device_mappings {
    device_name = "/dev/sda1"
    ebs {
      volume_size = 100
      volume_type = "gp3"
    }
  }

  # Bootstrap script that runs on first boot
  user_data = base64encode(templatefile("${path.module}/user_data_etl.sh", {
    docker_image         = var.etl_docker_image
    database_url         = var.database_url
    aws_region           = var.aws_region
  }))

  tags = {
    Name = "etl-ray-worker"
    ManagedBy = "terraform"
  }
}

resource "aws_instance" "etl" {
  launch_template {
    id      = aws_launch_template.etl.id
    version = "$Latest"
  }
  tags = { Name = "etl-ray-worker", ManagedBy = "terraform" }
}