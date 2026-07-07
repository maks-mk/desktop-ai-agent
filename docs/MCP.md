# MCP (Model Context Protocol)

`mcp.json` задаёт опциональные MCP-серверы. По умолчанию все серверы выключены, кроме `context7` (remote, read-only) — он включён из коробки.

## Policy

| `policy.read_only` | Поведение |
|---|---|
| `true` | Tool считается read-only, approval не требуется |
| `false` | Требует approval |
| не указан | Консервативный режим: approval по умолчанию |

## Пример подключения удалённого сервера

```json
{
  "context7": {
    "type": "remote",
    "url": "https://mcp.context7.com/mcp",
    "transport": "http",
    "enabled": true,
    "policy": {
      "read_only": true
    }
  }
}
```
