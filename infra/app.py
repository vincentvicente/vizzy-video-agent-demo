#!/usr/bin/env python3
import aws_cdk as cdk

from stack import VizzyStack

app = cdk.App()
VizzyStack(app, "VizzyStudio")
app.synth()
