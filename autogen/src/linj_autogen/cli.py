"""
LinJ Autogen CLI

命令行工具，用于独立运行 LinJ 工作流
"""

import asyncio
import json
import sys
from pathlib import Path

import click
import yaml

from .executor.runner import LinJExecutor, load_document


@click.group()
@click.version_option(version="0.1.0")
def cli():
    """LinJ Autogen - 执行 LinJ/ContiText 工作流"""
    pass


@cli.command()
@click.argument("document", type=click.Path(exists=True))
@click.option(
    "--state", "-s",
    type=click.Path(exists=True),
    help="初始状态 JSON 文件"
)
@click.option(
    "--output", "-o",
    type=click.Path(),
    help="输出结果到文件"
)
@click.option(
    "--format", "fmt",
    type=click.Choice(["json", "yaml"]),
    default="json",
    help="输出格式"
)
def run(document, state, output, fmt):
    """执行 LinJ 文档"""
    async def _run():
        # 加载文档
        doc = load_document(document)
        
        # 加载初始状态
        initial_state = {}
        if state:
            with open(state, 'r', encoding='utf-8') as f:
                if state.endswith('.yaml') or state.endswith('.yml'):
                    initial_state = yaml.safe_load(f)
                else:
                    initial_state = json.load(f)
        
        # 执行
        executor = LinJExecutor()
        result = await executor.run(doc, initial_state)
        
        # 格式化输出
        if fmt == "json":
            output_str = json.dumps(result, indent=2, ensure_ascii=False)
        else:
            output_str = yaml.dump(result, allow_unicode=True)
        
        # 输出
        if output:
            with open(output, 'w', encoding='utf-8') as f:
                f.write(output_str)
            click.echo(f"结果已保存到: {output}")
        else:
            click.echo(output_str)
    
    asyncio.run(_run())


@cli.command()
@click.argument("document", type=click.Path(exists=True))
def validate(document):
    """验证 LinJ 文档"""
    try:
        doc = load_document(document)
        
        # 检查引用
        errors = doc.validate_references()
        if errors:
            click.echo("验证失败:", err=True)
            for error in errors:
                click.echo(f"  - {error}", err=True)
            sys.exit(1)
        
        # 检查循环约束
        errors = doc.validate_loop_constraints()
        if errors:
            click.echo("警告:", err=True)
            for error in errors:
                click.echo(f"  - {error}", err=True)
        
        click.echo(f"文档验证通过: {document}")
        click.echo(f"版本: {doc.linj_version}")
        click.echo(f"节点数: {len(doc.nodes)}")
        click.echo(f"边数: {len(doc.edges)}")
        
    except Exception as e:
        click.echo(f"验证错误: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.argument("document", type=click.Path(exists=True))
def show(document):
    """显示 LinJ 文档结构"""
    doc = load_document(document)
    
    click.echo(f"文档: {document}")
    click.echo(f"版本: {doc.linj_version}")
    click.echo()
    
    click.echo("节点:")
    for node in doc.nodes:
        click.echo(f"  - [{node.type}] {node.id}")
        if node.title:
            click.echo(f"      标题: {node.title}")
    
    click.echo()
    click.echo("依赖边:")
    for edge in doc.edges:
        click.echo(f"  - {edge.from_} -> {edge.to} ({edge.kind})")


def main():
    """CLI 入口"""
    cli()


if __name__ == "__main__":
    main()
