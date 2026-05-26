"""
CDK Stack: Single EC2 instance running Vizzy Studio in Docker.

Resources:
- VPC (default)
- Security Group (8501 + SSH)
- EC2 instance (t3.small, Amazon Linux 2023)
- Elastic IP (stable address)
- Secrets in SSM Parameter Store
"""

GITHUB_REPO = "https://github.com/vincentvicente/vizzy-video-agent-demo"

import aws_cdk as cdk
from aws_cdk import (
    Stack,
    CfnOutput,
    aws_ec2 as ec2,
    aws_iam as iam,
)
from constructs import Construct


USER_DATA_SCRIPT = """#!/bin/bash
set -euo pipefail

# Install Docker
dnf update -y
dnf install -y docker git
systemctl enable docker
systemctl start docker

# Install Docker Compose plugin
mkdir -p /usr/local/lib/docker/cli-plugins
curl -SL "https://github.com/docker/compose/releases/latest/download/docker-compose-linux-$(uname -m)" \
    -o /usr/local/lib/docker/cli-plugins/docker-compose
chmod +x /usr/local/lib/docker/cli-plugins/docker-compose

# Create app directory and persistent data volume
mkdir -p /opt/vizzy/data

# Clone the repo (or pull updates on restart)
cd /opt/vizzy
if [ ! -d "app" ]; then
    git clone https://github.com/__REPO__ app
else
    cd app && git pull && cd ..
fi

# Write env file from SSM parameters
aws ssm get-parameter --name /vizzy/env-file --with-decryption --query 'Parameter.Value' --output text --region $(ec2-metadata --availability-zone | sed 's/.$//' | awk '{print $2}') > /opt/vizzy/app/.env

# Build and run
cd /opt/vizzy/app
docker build -t vizzy .
docker rm -f vizzy 2>/dev/null || true
docker run -d \
    --name vizzy \
    --restart unless-stopped \
    -p 8501:8501 \
    -v /opt/vizzy/data:/app/data \
    --env-file .env \
    vizzy
""".replace("__REPO__", GITHUB_REPO)


class VizzyStack(Stack):
    def __init__(self, scope: Construct, construct_id: str, **kwargs) -> None:
        super().__init__(scope, construct_id, **kwargs)

        # Use default VPC to keep things simple
        vpc = ec2.Vpc.from_lookup(self, "DefaultVpc", is_default=True)

        # Security group: allow 8501 (Streamlit) and 22 (SSH)
        sg = ec2.SecurityGroup(
            self, "VizzySG",
            vpc=vpc,
            description="Vizzy Studio - Streamlit + SSH",
            allow_all_outbound=True,
        )
        sg.add_ingress_rule(
            ec2.Peer.any_ipv4(),
            ec2.Port.tcp(8501),
            "Streamlit UI",
        )
        sg.add_ingress_rule(
            ec2.Peer.any_ipv4(),
            ec2.Port.tcp(22),
            "SSH access",
        )

        # IAM role so the instance can read SSM parameters
        role = iam.Role(
            self, "VizzyInstanceRole",
            assumed_by=iam.ServicePrincipal("ec2.amazonaws.com"),
            managed_policies=[
                iam.ManagedPolicy.from_aws_managed_policy_name(
                    "AmazonSSMManagedInstanceCore"
                ),
            ],
        )
        role.add_to_policy(iam.PolicyStatement(
            actions=["ssm:GetParameter"],
            resources=[
                f"arn:aws:ssm:{self.region}:{self.account}:parameter/vizzy/*"
            ],
        ))

        # EC2 instance
        instance = ec2.Instance(
            self, "VizzyInstance",
            vpc=vpc,
            instance_type=ec2.InstanceType("t3.small"),
            machine_image=ec2.MachineImage.latest_amazon_linux2023(),
            security_group=sg,
            role=role,
            vpc_subnets=ec2.SubnetSelection(subnet_type=ec2.SubnetType.PUBLIC),
            block_devices=[
                ec2.BlockDevice(
                    device_name="/dev/xvda",
                    volume=ec2.BlockDeviceVolume.ebs(
                        30, volume_type=ec2.EbsDeviceVolumeType.GP3
                    ),
                )
            ],
            user_data=ec2.UserData.custom(USER_DATA_SCRIPT),
        )

        # Elastic IP for a stable address
        eip = ec2.CfnEIP(self, "VizzyEIP")
        ec2.CfnEIPAssociation(
            self, "VizzyEIPAssoc",
            eip=eip.ref,
            instance_id=instance.instance_id,
        )

        CfnOutput(self, "AppURL", value=f"http://{eip.ref}:8501")
        CfnOutput(self, "InstanceId", value=instance.instance_id)
        CfnOutput(self, "PublicIP", value=eip.ref)
