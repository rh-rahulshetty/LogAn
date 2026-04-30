import sys


def serve():
    """Entry point for the MCP server (console_scripts: logan-mcp)."""
    from logan.mcp.server import mcp

    transport = "stdio"
    for i, arg in enumerate(sys.argv[1:], 1):
        if arg == "--transport" and i < len(sys.argv) - 1:
            transport = sys.argv[i + 1]
            break
        if arg.startswith("--transport="):
            transport = arg.split("=", 1)[1]
            break

    mcp.run(transport=transport)
