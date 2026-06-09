# Terraform — RAG API on AWS ECS Fargate

Deploys the FastAPI RAG service behind an Application Load Balancer with:

- VPC (2 AZs), public subnets (ALB), private subnets (ECS), NAT gateway
- ECR repository for container images
- ECS Fargate service (default: 2 tasks, 0.5 vCPU / 1 GB each)
- Secrets Manager for `OPENAI_API_KEY`, `PINECONE_API_KEY`, `API_KEY`
- CloudWatch Logs at `/ecs/rag-api-prod` (name varies by environment)

## Prerequisites

- AWS CLI configured (`aws configure`)
- Terraform >= 1.5
- Docker (for building and pushing the image)
- Pinecone index created (`python scripts/ensure_pinecone_index.py`)

## Initial deploy

```bash
cd infra/terraform
cp terraform.tfvars.example terraform.tfvars

export TF_VAR_openai_api_key="sk-..."
export TF_VAR_pinecone_api_key="..."
export TF_VAR_api_key="your-secure-api-key"

terraform init
terraform plan
terraform apply
```

Note the outputs:

```bash
terraform output alb_url
terraform output ecr_repository_url
```

## Push container image and roll ECS

From the repository root:

```bash
./scripts/deploy.sh
```

Or manually:

```bash
AWS_REGION=us-east-1
ECR_URL=$(terraform -chdir=infra/terraform output -raw ecr_repository_url)
CLUSTER=$(terraform -chdir=infra/terraform output -raw ecs_cluster_name)
SERVICE=$(terraform -chdir=infra/terraform output -raw ecs_service_name)

aws ecr get-login-password --region "$AWS_REGION" \
  | docker login --username AWS --password-stdin "${ECR_URL%/*}"

docker build -t rag-api .
docker tag rag-api:latest "$ECR_URL:latest"
docker push "$ECR_URL:latest"

aws ecs update-service \
  --region "$AWS_REGION" \
  --cluster "$CLUSTER" \
  --service "$SERVICE" \
  --force-new-deployment
```

## Verify

```bash
ALB_URL=$(terraform -chdir=infra/terraform output -raw alb_url)
curl "$ALB_URL/health"
```

## HTTPS (optional)

1. Request or import an ACM certificate for your domain in the same region.
2. Set `certificate_arn` in `terraform.tfvars`.
3. Run `terraform apply`.
4. Point your domain's DNS A/ALIAS record at the ALB DNS name.

When `certificate_arn` is set, HTTP traffic on port 80 redirects to HTTPS.

## Remote state (recommended)

Uncomment the `backend "s3"` block in [`main.tf`](main.tf) and create:

- S3 bucket for state
- DynamoDB table for state locking

## Teardown

```bash
cd infra/terraform
terraform destroy
```

Ensure the ECR repository is empty or set `force_delete = true` (already enabled in this module).

## Deferred

- WAF, auto-scaling, multi-region, S3 for full chunk text storage
