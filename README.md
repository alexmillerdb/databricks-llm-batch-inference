# Databricks LLM batch inference with FM APIs, Provisioned Throughput, & External Models

## Overview
The goal of this project is to simplify batch inference with LLMs through simple API interface for Databricks GenAI model serving. This follows a simple process:

1. Create a Delta table that includes the text you want to run inference on.
2. Define the data configurations such as table path
3. Define the inference congirations such as endpoint, prompt, concurrency, and LLM task (completion or task)
4. Run the batch inference based on configurations

## Features
- Simplified API that allows you to iterate quickly without having to worry about writing Python async or Spark UDF commands
- Built-in retry logic on the client-side for specific error status codes such as 429 and 503
- Simple developer experience that allows users to test and measure different prompts and concurrent requests to meet SLAs (throughput/latency) and model performance

## Installation

## Notice

## Supported Frameworks

## Running LLM batch inference using Python 

## Running LLM batch inference using Spark
