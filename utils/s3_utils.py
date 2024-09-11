import pickle
import boto3
import pandas as pd
import os
from dotenv import load_dotenv
from io import StringIO
import rioxarray as rxr
import geopandas as gpd
import s3fs
import xarray as xr

load_dotenv()

access_token=os.getenv('aws_access_key')
access_token_secret=os.getenv('aws_secret_access_key')
    
def getS3Files(bucket):
    output = []
    session = boto3.Session(aws_access_key_id=access_token,
                          aws_secret_access_key=access_token_secret)
    
    s = session.client('s3')
    paginator = s.get_paginator('list_objects_v2')
    pages = paginator.paginate(Bucket=bucket)

    for page in pages:
        #for key in s.list_objects(Bucket=bucket)['Contents']:
        for key in page['Contents']:
            output.append(key['Key'])
    return output

def writePklToS3(bucket, filename, data):
    pickle_byte_obj = pickle.dumps(data) 
    session = boto3.Session(aws_access_key_id=access_token,
                          aws_secret_access_key=access_token_secret)
    resource = session.resource('s3')
    
    resource.Object(bucket,filename + '.pkl').put(Body=pickle_byte_obj)
    
def write2S3(data, filename, bucket):
    session = boto3.Session(aws_access_key_id=access_token,
                          aws_secret_access_key=access_token_secret)
    
    s3 = session.resource('s3')
    object = s3.Object(bucket, filename)
    object.put(Body=data)

def writecsv2S3(data, filename, bucket):
    session = boto3.Session(aws_access_key_id=access_token,
                          aws_secret_access_key=access_token_secret)
    csv_buffer = StringIO()
    data.to_csv(csv_buffer)
    s3 = session.resource('s3')
    object = s3.Object(bucket, filename)
    object.put(Body=csv_buffer.getvalue())

def rasterFromS3(filename,bucket):
    fs_s3 = s3fs.S3FileSystem(anon=False, 
                        key=access_token, 
                        secret=access_token_secret) 
                        #token=temp_creds_req['sessionToken'])
    s3_url = 's3://'+bucket+'/'+filename
    s3_file_obj = fs_s3.open(s3_url, mode='rb')
    ds = rxr.open_rasterio(s3_file_obj, chuncks=True)
    return ds
 
def readNCFromS3(filename,bucket):
    fs_s3 = s3fs.S3FileSystem(anon=False, 
                          key=access_token, 
                          secret=access_token_secret) 
                          #token=temp_creds_req['sessionToken'])
    s3_url = 's3://'+bucket+'/'+filename
    s3_file_obj = fs_s3.open(s3_url, mode='rb')
    ds = xr.open_dataset(s3_file_obj, engine='h5netcdf')
    return ds
    
def readSHPfromS3(filename, bucket):
    #this function needs some work, currently is not working. Use local copies. 

    # session = boto3.Session(aws_access_key_id=access_token,
    #                       aws_secret_access_key=access_token_secret,
    #                       region_name='us-east-2')
    # with fiona.Env(session=AWSSession(session)):
    #     return gpd.read_file('s3://'+bucket+'/'+'filename') #filename must have a .zip label
    fs_s3 = s3fs.S3FileSystem(anon=False, 
                        key=access_token, 
                        secret=access_token_secret) 
                        #token=temp_creds_req['sessionToken'])
    s3_url = 's3://zip//'+bucket+'/'+filename
    s3_file_obj = fs_s3.open(s3_url, mode='rb')
    ds = gpd.read_file(s3_file_obj)
    return ds

def readParquetFromS3(filename, bucket, engine, filter):
    # fs_s3 = s3fs.S3FileSystem(anon=False, 
    #                       key=access_token, 
    #                       secret=access_token_secret) 
    #                       #token=temp_creds_req['sessionToken'])
    s3_url = f's3://{bucket}/{filename}'
    df = pd.read_parquet(s3_url, engine='pyarrow', filters=filter)                    
    # dataset = pq.ParquetDataset(s3_url, filesystem=fs_s3)
    # df = dataset.read_pandas().to_pandas()
    return df

def writeParquetToS3(df, filename, bucket, col):
    # fs_s3 = s3fs.S3FileSystem(anon=False, 
    #                       key=access_token, 
    #                       secret=access_token_secret) 
    #                       #token=temp_creds_req['sessionToken'])
    s3_url = f's3://{bucket}/{filename}'
    # with fs_s3.open(s3_url, 'wb') as f:
    df.to_parquet(s3_url, compression ='snappy', partition_cols = col)

    # table = pa.Table.from_pandas(df)
    # pq.write_to_dataset(table=table, 
    #                 root_path=s3_url,
    #                 filesystem=fs_s3,
    #                 compression='snappy') 

def readPklFromS3(bucket, filename):
    session = boto3.Session(aws_access_key_id=access_token,
                          aws_secret_access_key=access_token_secret)
    
    s = session.client('s3')
    try:
        obj = s.get_object(Bucket = bucket, Key = filename)
    except s.exceptions.NoSuchKey:
        return {}
    return pickle.loads(obj['Body'].read())

def readCSVFromS3(bucket, filename, sep=',', doublequote=True, header=0,skip_blank_lines=False, encoding='utf-8'):
    session = boto3.Session(aws_access_key_id=access_token,
                          aws_secret_access_key=access_token_secret)
    
    s = session.client('s3')
    response = s.get_object(Bucket=bucket, Key=filename)
    data = StringIO(response['Body'].read().decode(encoding))
    return(pd.read_csv(data, sep=sep, encoding=encoding, doublequote=doublequote, header=header, skip_blank_lines=skip_blank_lines))

def readTxtFromS3(bucket, filename, encoding="utf-8"):
    session = boto3.Session(aws_access_key_id=access_token,
                          aws_secret_access_key=access_token_secret)
    
    s = session.client('s3')
    obj = s.get_object(Bucket = bucket, Key = filename)
    return obj['Body'].read().decode(encoding)

def deleteFromS3(bucket, filename):
    session = boto3.Session(aws_access_key_id=access_token,
                          aws_secret_access_key=access_token_secret)
    resource = session.resource('s3')
    resource.Object(bucket, filename).delete()