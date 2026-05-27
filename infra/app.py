#!/usr/bin/env python3
import os
import aws_cdk as cdk

from stack import VizzyStack

app = cdk.App()
VizzyStack(
    app,
    "VizzyStudio",
    env=cdk.Environment(
        account=os.environ["CDK_DEFAULT_ACCOUNT"],
        region=os.environ["CDK_DEFAULT_REGION"],
    ),
)
app.synth()
