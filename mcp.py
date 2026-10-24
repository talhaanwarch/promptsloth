"""Minimal MCP JSON-RPC client for AcruxCore (requests only)."""
import json, os, requests

BASE = os.environ.get("ACRUXCORE_BASE_URL", "https://api.acruxcore.com/api/v1")
KEY = os.environ["ACRUXCORE_API_KEY"]
_S = requests.Session()


def call(method, params=None, rpc_id=1):
    r = _S.post(BASE + "/mcp", timeout=300, json={"jsonrpc": "2.0", "id": rpc_id, "method": method, "params": params or {}},
                headers={"Authorization": f"Bearer {KEY}", "Accept": "application/json, text/event-stream",
                         "MCP-Protocol-Version": "2025-06-18"})
    r.raise_for_status()
    for line in r.text.splitlines():
        if line.startswith("data:"):
            return json.loads(line[5:])
    return r.json()


def tool(name, args):
    """Call one MCP tool; returns (text, is_error)."""
    obj = call("tools/call", {"name": name, "arguments": args}, 99)
    if "error" in obj:
        return json.dumps(obj["error"]), True
    res = obj["result"]
    text = "\n".join(p.get("text", "") for p in res.get("content", []) if p.get("type") == "text")
    return text, bool(res.get("isError"))


def tool_json(name, args):
    """First JSON value in a tool's text result."""
    text, err = tool(name, args)
    if err:
        raise RuntimeError(f"{name}: {text[:500]}")
    for i, ch in enumerate(text):
        if ch in "{[":
            try:
                return json.JSONDecoder().raw_decode(text[i:])[0]
            except json.JSONDecodeError:
                continue
    return {"_text": text}
