from setuptools import setup, find_packages

setup(
    name="llm-batch-inference",
    author="Alex Miller",
    author_email="alex.miller@databricks.com",
    description="LLM Batch Inference for Databricks FM APIs/Provisioned Throughput",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/alexmillerdb/databricks-llm-batch-inference",
    version="0.1.1",
    packages=find_packages(include=['llm_batch_inference', 'llm_batch_inference.*']),
    install_requires=[
        "httpx==0.27.0",
        "mlflow-skinny[databricks]",
        "tenacity==8.2.3",
        "openai",
        "pydantic",
        "databricks-sdk",
        "asyncio",
        "nest-asyncio"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.10"
)