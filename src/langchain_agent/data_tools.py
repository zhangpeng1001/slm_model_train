"""
数据平台工具集
为LangChain Agent提供数据操作工具
"""
import json
import time
from typing import Dict, Any, List
from langchain.tools import BaseTool
from pydantic import BaseModel, Field
import logging

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataQueryInput(BaseModel):
    """数据查询工具输入"""
    data_name: str = Field(description="要查询的数据名称")
    query_type: str = Field(description="查询类型：schema/count/sample", default="schema")

class DataCollectionInput(BaseModel):
    """数据采集工具输入"""
    data_name: str = Field(description="要采集的数据名称")
    source_type: str = Field(description="数据源类型：database/api/file", default="database")
    collection_config: Dict[str, Any] = Field(description="采集配置", default_factory=dict)

class DataStorageInput(BaseModel):
    """数据入库工具输入"""
    data_name: str = Field(description="要入库的数据名称")
    data_content: str = Field(description="数据内容或数据路径")
    storage_config: Dict[str, Any] = Field(description="存储配置", default_factory=dict)

class DataServiceInput(BaseModel):
    """数据服务工具输入"""
    data_name: str = Field(description="要发布服务的数据名称")
    service_type: str = Field(description="服务类型：rest_api/graphql/websocket", default="rest_api")
    service_config: Dict[str, Any] = Field(description="服务配置", default_factory=dict)

class DataQueryTool(BaseTool):
    """数据查询工具"""
    name: str = "data_query"
    description: str = "查询指定数据的基本信息，包括数据结构、数据量、样本数据等"
    args_schema: type = DataQueryInput

    def _run(self, data_name: str, query_type: str = "schema") -> str:
        """执行数据查询"""
        try:
            logger.info(f"开始查询数据: {data_name}, 查询类型: {query_type}")
            
            # 模拟数据查询过程
            time.sleep(1)  # 模拟查询耗时
            
            if query_type == "schema":
                result = {
                    "data_name": data_name,
                    "schema": {
                        "columns": ["id", "name", "create_time", "status"],
                        "types": ["int", "varchar", "datetime", "int"],
                        "description": f"{data_name}数据表结构信息"
                    },
                    "table_size": "约10万条记录",
                    "last_update": "2024-01-13 09:30:00"
                }
            elif query_type == "count":
                result = {
                    "data_name": data_name,
                    "total_count": 98765,
                    "active_count": 87654,
                    "last_count_time": "2024-01-13 09:30:00"
                }
            elif query_type == "sample":
                result = {
                    "data_name": data_name,
                    "sample_data": [
                        {"id": 1, "name": "示例数据1", "create_time": "2024-01-01", "status": 1},
                        {"id": 2, "name": "示例数据2", "create_time": "2024-01-02", "status": 1},
                        {"id": 3, "name": "示例数据3", "create_time": "2024-01-03", "status": 0}
                    ],
                    "sample_size": 3
                }
            else:
                result = {"error": f"不支持的查询类型: {query_type}"}
            
            logger.info(f"数据查询完成: {data_name}")
            return json.dumps(result, ensure_ascii=False, indent=2)
            
        except Exception as e:
            error_msg = f"数据查询失败: {str(e)}"
            logger.error(error_msg)
            return json.dumps({"error": error_msg}, ensure_ascii=False)

    async def _arun(self, data_name: str, query_type: str = "schema") -> str:
        """异步执行数据查询"""
        return self._run(data_name, query_type)

class DataCollectionTool(BaseTool):
    """数据采集工具"""
    name: str = "data_collection"
    description: str = "从指定数据源采集数据，支持数据库、API、文件等多种数据源"
    args_schema: type = DataCollectionInput

    def _run(self, data_name: str, source_type: str = "database", collection_config: Dict[str, Any] = None) -> str:
        """执行数据采集"""
        try:
            if collection_config is None:
                collection_config = {}
                
            logger.info(f"开始采集数据: {data_name}, 数据源类型: {source_type}")
            
            # 模拟数据采集过程
            time.sleep(2)  # 模拟采集耗时
            
            result = {
                "data_name": data_name,
                "source_type": source_type,
                "collection_status": "success",
                "collected_records": 1234,
                "collection_time": "2024-01-13 09:35:00",
                "data_file_path": f"/data/collected/{data_name}_{int(time.time())}.json",
                "collection_config": collection_config,
                "data_preview": [
                    {"id": 1001, "name": f"{data_name}_record_1", "value": 100.5},
                    {"id": 1002, "name": f"{data_name}_record_2", "value": 200.8},
                    {"id": 1003, "name": f"{data_name}_record_3", "value": 150.3}
                ]
            }
            
            logger.info(f"数据采集完成: {data_name}, 采集记录数: {result['collected_records']}")
            return json.dumps(result, ensure_ascii=False, indent=2)
            
        except Exception as e:
            error_msg = f"数据采集失败: {str(e)}"
            logger.error(error_msg)
            return json.dumps({"error": error_msg}, ensure_ascii=False)

    async def _arun(self, data_name: str, source_type: str = "database", collection_config: Dict[str, Any] = None) -> str:
        """异步执行数据采集"""
        return self._run(data_name, source_type, collection_config)

class DataStorageTool(BaseTool):
    """数据入库工具"""
    name: str = "data_storage"
    description: str = "将采集的数据存储到数据仓库中，支持多种存储格式和配置"
    args_schema: type = DataStorageInput

    def _run(self, data_name: str, data_content: str, storage_config: Dict[str, Any] = None) -> str:
        """执行数据入库"""
        try:
            if storage_config is None:
                storage_config = {}
                
            logger.info(f"开始数据入库: {data_name}")
            
            # 模拟数据入库过程
            time.sleep(1.5)  # 模拟入库耗时
            
            result = {
                "data_name": data_name,
                "storage_status": "success",
                "storage_location": f"warehouse.{data_name}_table",
                "stored_records": 1234,
                "storage_time": "2024-01-13 09:37:00",
                "storage_config": storage_config,
                "table_info": {
                    "database": "data_warehouse",
                    "table": f"{data_name}_table",
                    "partition": "dt=2024-01-13",
                    "storage_format": "parquet",
                    "compression": "snappy"
                },
                "data_quality_check": {
                    "null_count": 0,
                    "duplicate_count": 2,
                    "data_quality_score": 98.5
                }
            }
            
            logger.info(f"数据入库完成: {data_name}, 入库记录数: {result['stored_records']}")
            return json.dumps(result, ensure_ascii=False, indent=2)
            
        except Exception as e:
            error_msg = f"数据入库失败: {str(e)}"
            logger.error(error_msg)
            return json.dumps({"error": error_msg}, ensure_ascii=False)

    async def _arun(self, data_name: str, data_content: str, storage_config: Dict[str, Any] = None) -> str:
        """异步执行数据入库"""
        return self._run(data_name, data_content, storage_config)

class DataServiceTool(BaseTool):
    """数据服务发布工具"""
    name: str = "data_service"
    description: str = "将数据封装为API服务对外提供，支持REST API、GraphQL等多种服务类型"
    args_schema: type = DataServiceInput

    def _run(self, data_name: str, service_type: str = "rest_api", service_config: Dict[str, Any] = None) -> str:
        """执行数据服务发布"""
        try:
            if service_config is None:
                service_config = {}
                
            logger.info(f"开始发布数据服务: {data_name}, 服务类型: {service_type}")
            
            # 模拟服务发布过程
            time.sleep(2)  # 模拟发布耗时
            
            service_port = 8080 + hash(data_name) % 1000
            
            result = {
                "data_name": data_name,
                "service_type": service_type,
                "service_status": "published",
                "service_url": f"http://localhost:{service_port}/api/v1/{data_name}",
                "service_endpoints": {
                    "query": f"GET /api/v1/{data_name}",
                    "create": f"POST /api/v1/{data_name}",
                    "update": f"PUT /api/v1/{data_name}/{{id}}",
                    "delete": f"DELETE /api/v1/{data_name}/{{id}}"
                },
                "service_config": service_config,
                "publish_time": "2024-01-13 09:40:00",
                "service_info": {
                    "version": "v1.0.0",
                    "rate_limit": "1000 requests/hour",
                    "authentication": "API Key required",
                    "documentation": f"http://localhost:{service_port}/docs"
                },
                "health_check": {
                    "status": "healthy",
                    "response_time": "< 100ms",
                    "uptime": "100%"
                }
            }
            
            logger.info(f"数据服务发布完成: {data_name}, 服务地址: {result['service_url']}")
            return json.dumps(result, ensure_ascii=False, indent=2)
            
        except Exception as e:
            error_msg = f"数据服务发布失败: {str(e)}"
            logger.error(error_msg)
            return json.dumps({"error": error_msg}, ensure_ascii=False)

    async def _arun(self, data_name: str, service_type: str = "rest_api", service_config: Dict[str, Any] = None) -> str:
        """异步执行数据服务发布"""
        return self._run(data_name, service_type, service_config)

# 工具列表
DATA_TOOLS = [
    DataQueryTool(),
    DataCollectionTool(),
    DataStorageTool(),
    DataServiceTool()
]

def get_tool_descriptions() -> str:
    """获取所有工具的描述信息"""
    descriptions = []
    for tool in DATA_TOOLS:
        descriptions.append(f"- {tool.name}: {tool.description}")
    return "\n".join(descriptions)

def get_tool_by_name(tool_name: str) -> BaseTool:
    """根据名称获取工具"""
    for tool in DATA_TOOLS:
        if tool.name == tool_name:
            return tool
    return None
