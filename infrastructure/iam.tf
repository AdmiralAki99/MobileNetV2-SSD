# ---- IAM Role ----
resource "aws_iam_role" "training" {
  name = "ml-training-role"

  # This policy says: "EC2 instances are allowed to assume this role"
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "ec2.amazonaws.com"
        }
      }
    ]
  })

  tags = {
    ManagedBy = "terraform"
  }
}

# ---- IAM Policy ----
resource "aws_iam_role_policy" "s3_access" {
  name = "ml-training-s3-access"
  role = aws_iam_role.training.id  # Attach to the role above

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        # Allow read/write to checkpoint bucket
        Effect = "Allow"
        Action = [
          "s3:GetObject",
          "s3:PutObject",
          "s3:ListBucket",
          "s3:DeleteObject"
        ]
        Resource = [
          "arn:aws:s3:::${var.s3_checkpoint_bucket}",     # The bucket itself
          "arn:aws:s3:::${var.s3_checkpoint_bucket}/*"     # All objects in it
        ]
      },
      {
        # Allow read/write to artifacts bucket (final weights, saved models, exports)
        Effect = "Allow"
        Action = [
          "s3:GetObject",
          "s3:PutObject",
          "s3:ListBucket",
          "s3:DeleteObject"
        ]
        Resource = [
          "arn:aws:s3:::${var.s3_artifacts_bucket}",
          "arn:aws:s3:::${var.s3_artifacts_bucket}/*"
        ]
      },
      {
        # Allow read-only access to dataset bucket
        Effect = "Allow"
        Action = [
          "s3:GetObject",
          "s3:PutObject",
          "s3:ListBucket"
        ]
        Resource = [
          "arn:aws:s3:::${var.s3_dataset_bucket}",
          "arn:aws:s3:::${var.s3_dataset_bucket}/*"
        ]
      },
      {
        # Allow read-only access to dataset bucket
        Effect = "Allow"
        Action = [
          "s3:GetObject",
          "s3:PutObject",
          "s3:ListBucket"
        ]
        Resource = [
          "arn:aws:s3:::${var.experiment_bucket}",
          "arn:aws:s3:::${var.experiment_bucket}/*"
        ]
      }
    ]
  })
}

# DynamoDB resource
resource "aws_iam_role_policy" "dynamodb_access" {
  name = "ml-training-dynamodb-access"
  role = aws_iam_role.training.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "dynamodb:GetItem",
          "dynamodb:PutItem",
          "dynamodb:UpdateItem",
          "dynamodb:Query",
          "dynamodb:BatchWriteItem",
        ]
        Resource = [
          data.aws_dynamodb_table.experiment_ledger.arn,
          "${data.aws_dynamodb_table.experiment_ledger.arn}/index/*",
        ]
      }
    ]
  })
}

# ---- Instance Profile ----
resource "aws_iam_instance_profile" "training" {
  name = "ml-training-profile"
  role = aws_iam_role.training.name
}
