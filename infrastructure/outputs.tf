output "instance_public_ip" {
  description = "Public IP of the training instance"
  value       = try(data.aws_instances.training.public_ips[0], "pending")
}

output "instance_id" {
  description = "EC2 instance ID"
  value       = try(data.aws_instances.training.ids[0], "pending")
}

output "ssh_command" {
  description = "SSH command to connect to the instance"
  value       = "ssh -i ~/.ssh/${var.key_pair_name}.pem ubuntu@${try(data.aws_instances.training.public_ips[0], "pending")}"
}

output "tensorboard_url" {
  description = "TensorBoard URL (after training starts)"
  value       = "http://${try(data.aws_instances.training.public_ips[0], "pending")}:6006"
}

output "ami_used" {
  description = "AMI that was selected"
  value       = data.aws_ami.deep_learning.name
}
