# Vizzy Studio — AWS Deployment

Single EC2 instance running the Streamlit app in Docker, managed via AWS CDK.

## Prerequisites

- [AWS CLI](https://docs.aws.amazon.com/cli/latest/userguide/install-cliv2.html) configured (`aws configure`)
- [AWS CDK CLI](https://docs.aws.amazon.com/cdk/v2/guide/getting-started.html#getting-started-install) (`npm install -g aws-cdk`)
- [uv](https://docs.astral.sh/uv/) installed

## First-time Setup

```bash
cd infra

# Install dependencies
uv sync

# Bootstrap CDK in your AWS account (one-time)
cdk bootstrap
```

## Store Your Secrets

Before deploying, store the `.env` contents in SSM Parameter Store:

```bash
aws ssm put-parameter \
    --name /vizzy/env-file \
    --type SecureString \
    --value "$(cat ../.env)"
```

## Deploy

```bash
cd infra
cdk deploy
```

CDK will show the resources to be created and ask for confirmation. After ~3-5 minutes, it outputs:

```
VizzyStudio.AppURL = http://<elastic-ip>:8501
VizzyStudio.PublicIP = <elastic-ip>
```

## Update the App

SSH into the instance and pull the latest code:

```bash
ssh ec2-user@<elastic-ip>
cd /opt/vizzy/app
git pull
docker build -t vizzy .
docker rm -f vizzy
docker run -d --name vizzy --restart unless-stopped -p 8501:8501 -v /opt/vizzy/data:/app/data --env-file .env vizzy
```

Or terminate and redeploy (user-data re-runs on new instances):

```bash
cdk deploy
```

## Tear Down

```bash
cd infra
cdk destroy
```

This removes all AWS resources (instance, EIP, security group). The `data/` directory on the instance is lost.

## Cost

- t3.small: ~$15/mo (24/7)
- Elastic IP: ~$4/mo
- EBS 30GB gp3: ~$2.50/mo
- **Total: ~$21/mo**

## Architecture

```
Internet :8501 → Elastic IP → EC2 (t3.small, Amazon Linux 2023)
                                 └─ Docker: Streamlit + ffmpeg
                                 └─ /opt/vizzy/data (persistent on EBS)
                                 └─ .env from SSM Parameter Store
```
