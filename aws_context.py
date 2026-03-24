"""
aws_context.py
Central AWS session / client manager for the project.
All modules should use this instead of creating boto3 clients directly.
"""
from typing import Optional

import boto3
import s3fs
 
 
_REGION = "eu-west-1"
 
_session: Optional[boto3.session.Session] = None
_rt = None
_s3 = None
_sts = None
_s3fs = None
 
 
def set_session(session: boto3.session.Session) -> None:
    global _session, _rt, _s3, _sts, _s3fs
 
    _session = session
    _rt = None
    _s3 = None
    _sts = None
    _s3fs = None
    
 
 
def create_session(profile_name: Optional[str] = None, region_name: str = "eu-west-1"):
    session = boto3.Session(
        profile_name=profile_name,
        region_name=region_name,
    )
    set_session(session)
    return session
 
 
def get_session():
    global _session
    if _session is None:
        _session = boto3.Session(region_name=_REGION)
    return _session
 
 
def get_bedrock_runtime():
    global _rt
    if _rt is None:
        _rt = get_session().client("bedrock-runtime", region_name=_REGION)
    return _rt
 
 
def get_s3():
    global _s3
    if _s3 is None:
        _s3 = get_session().client("s3", region_name=_REGION)
    return _s3
 
 
def get_sts():
    global _sts
    if _sts is None:
        _sts = get_session().client("sts", region_name=_REGION)
    return _sts
 
 
def get_s3fs():
    global _s3fs
 
    if _s3fs is None:
        creds = get_session().get_credentials().get_frozen_credentials()
        _s3fs = s3fs.S3FileSystem(
            anon=False,
            key=creds.access_key,
            secret=creds.secret_key,
            token=creds.token,
            client_kwargs={"region_name": _REGION},
        )
 
    return _s3fs
 
 
def get_caller_identity():
    return get_sts().get_caller_identity()