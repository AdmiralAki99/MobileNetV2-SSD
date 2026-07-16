data "aws_dynamodb_table" "experiment_ledger" {
  name = var.dynamodb_table_name
}