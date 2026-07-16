# A security group is like a firewall around your instance
resource "aws_security_group" "training" {
  name        = "ml-training-sg"
  description = "Security group for ML training instances"

  # Allow SSH from your IP
  ingress {
    description = "SSH access"
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = [var.allowed_ssh_cidr]  # Restrict to your IP in production
  }

  # Allow TensorBoard access
  ingress {
    description = "TensorBoard"
    from_port   = 6006
    to_port     = 6006
    protocol    = "tcp"
    cidr_blocks = [var.allowed_ssh_cidr]
  }

  # ---- Outbound Rules (what the instance can connect TO) ----

  egress {
    description = "All outbound traffic"
    from_port   = 0
    to_port     = 0
    protocol    = "-1"          # -1 = all protocols
    cidr_blocks = ["0.0.0.0/0"]
  }

  # Ray dashboard information
  ingress{
    description = "Ray Dashboard"
    from_port = 8265 # DAG polls port 8265 for Ray dashboard info
    to_port = 8265
    protocol = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  ingress{
    description = "Ray GCS"
    from_port = 6379
    to_port = 6379
    protocol = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Name      = "ml-training-sg"
    ManagedBy = "terraform"
  }
}

