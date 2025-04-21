import boto3
from botocore.exceptions import NoCredentialsError

# 替换为你的 bucket 名字和文件路径
bucket_name = 'emg-signal-storage'
file_path = 'upload/signal_1.csv'  # 你要上传的本地文件
s3_key = 'emg_data/emg_data.csv'  # 上传后在 S3 中的路径

def upload_to_aws(local_file, bucket, s3_file):
    s3 = boto3.client('s3')
    try:
        s3.upload_file(local_file, bucket, s3_file)
        print("✅ 上传成功")
    except FileNotFoundError:
        print("❌ 文件未找到")
    except NoCredentialsError:
        print("❌ AWS 认证失败，请检查 access key 配置")

upload_to_aws(file_path, bucket_name, s3_key)
