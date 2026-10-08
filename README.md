# agent-lab
[![CI](https://github.com/timo2553/agent-lab/actions/workflows/ci.yml/badge.svg)](https://github.com/timo2553/agent-lab/actions/workflows/ci.yml)

从零实现的电商客服 Agent 工程练习。不用 LangChain / LangGraph 这类编排框架，把
**会话落盘 → 上下文构建 → 检索 → 工具调用 → 工作流与人工审批 → 可观测** 这条链路自己写一遍，
每一步都留下测试和取舍记录。课程仓只作为需求参考，本仓代码全部自己实现。

当前进度：**第 02 步 模型接入与会话闭环已完成**，33 个测试全绿（`33 passed`），全部离线可跑。

## 为什么从会话持久化开始

课程里的 `session_state` 是进程内的字典：Agent 一重启，`message_count` 和会话上下文全部丢失。
这是真实问题，不是编出来的需求。所以第一步先把会话落盘、能读回、能按 `session_id` 隔离。

## 快速开始

```powershell
py -3 -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements.txt
.\.venv\Scripts\python -m pytest -v
```

期望结果：`33 passed`。测试全部跑在临时目录里，不在你机器上留垃圾文件。

跑起来看看（**不需要 API Key**，`--fake` 会用本地假模型走完整条链路）：

```powershell
.\.venv\Scripts\python -m agent_lab ask "耳机什么时候到货？" --session demo --fake
.\.venv\Scripts\python -m agent_lab chat --session demo --fake   # 连续对话，exit 退出
.\.venv\Scripts\python -m agent_lab usage --session demo          # 累计 token 与估算费用
```

接真实模型时配三个环境变量即可（默认接 DeepSeek 的 OpenAI 兼容接口）：

```powershell
$env:AGENT_LAB_API_KEY = "sk-..."          # 也可以用 DEEPSEEK_API_KEY
$env:AGENT_LAB_BASE_URL = "https://api.deepseek.com"
$env:AGENT_LAB_MODEL = "deepseek-chat"
.\.venv\Scripts\python -m agent_lab chat --session demo
```

## 目录结构

```
agent-lab/
├── agent_lab/
│   ├── models.py              # Message / UsageRecord
│   ├── session_store.py       # 会话与用量的持久化存储（SQLite）
│   ├── config.py              # 模型接入参数（从环境变量读）
│   ├── cost.py                # 价目表、成本估算、预算熔断
│   ├── cli.py                 # 命令行入口（ask / chat / usage）
│   ├── llm/
│   │   ├── client.py          # LLMClient 契约 + OpenAI 兼容实现（urllib）
│   │   ├── errors.py          # 错误分类：哪些能重试，哪些不能
│   │   ├── types.py           # Usage / LLMResponse
│   │   └── fake.py            # 假模型，测试与 --fake 演示都用它
│   └── chat/
│       └── service.py         # 最小对话闭环：读历史 -> 调模型 -> 落库
├── tests/
│   ├── test_session_store.py            # 主流程：写入 / 读回 / 隔离 / 限制条数 / 删除
│   ├── test_session_store_edges.py      # 自补边界：并发写、重复删除
│   ├── test_session_store_migration.py  # 老库迁移：补 status 列
│   ├── test_llm_client.py               # 请求构造、解析、错误分类与重试
│   ├── test_chat_service.py             # 历史回灌、失败落库、预算熔断
│   └── test_cost.py                     # 用量累加与成本估算
├── conftest.py                # 让 pytest 无论从哪个目录启动都能 import agent_lab
├── requirements.txt
└── README.md
```

## 第 01 步：会话持久化

对外只有三个方法，语义由测试定义（[tests/test_session_store.py](tests/test_session_store.py)）：

| 方法 | 语义 |
|---|---|
| `append_message(session_id, role, content) -> Message` | 追加一条消息，返回落库后的对象 |
| `load_history(session_id, limit=20) -> list[Message]` | 取最近 `limit` 条，但按时间正序返回；会话不存在返回空列表 |
| `delete_session(session_id) -> None` | 删除该会话全部消息，不影响其他会话；删不存在的会话不算错误 |

关键细节：换一个全新实例指向同一个文件（相当于进程重启）必须能读回历史，所以数据不能只放在内存里。

## 第 02 步：模型接入与会话闭环

一轮 `ask()` 的顺序是固定的：
**查预算 → 读历史 → 组 prompt → 落库用户输入 → 调模型 → 落库回复 → 记用量**。

| 模块 | 对外契约 |
|---|---|
| `LLMClient`（Protocol） | `complete(messages) -> LLMResponse`，入参类型只有 `Message`；真实实现和假模型都满足它，所以测试永远离线 |
| `OpenAICompatClient` | POST `/chat/completions`，默认 30s 超时；只重试 429 / 5xx / 超时 / 断网，退避 0.5s → 1s |
| `ChatService.ask(session_id, text)` | 返回 `Reply(text, usage, user_message, assistant_message)` |
| `ChatService.session_usage(session_id)` | 该会话累计用量与估算费用；从库里读，重启后依然准 |
| CLI | `ask` / `chat` / `usage`；`--fake` 免 Key，`--budget` 设单会话 token 上限 |

错误分类（`llm/errors.py`）不是装饰：`retryable` 决定了要不要重试，
也决定了 CLI 该提示用户"稍后再试"还是"请检查配置，或转人工"。

## 设计决策

> 每做一个取舍就记一段：**背景 / 选了什么 / 为什么 / 代价是什么**。
> 面试官问的是这些，而不是"你用了什么库"。

### 1. 存储形态：SQLite（不是 JSONL）

**背景**：需要一个能在进程重启后读回、能按 `session_id` 取最近 N 条、能支撑并发写的存储。

**选择**：SQLite。标准库自带 `sqlite3`，单文件，零额外依赖。

**为什么不是 JSONL**：JSONL 追加写最省事，但"取某会话最近 N 条"要全文件扫描，还得自己维护顺序——
而本题的核心语义恰恰就是这个（`load_history`）。SQLite 用 `(session_id, seq)` 一次搞定过滤和排序，
还自带事务。

**为什么不是 MySQL / Postgres**：本地练习没必要起服务、管连接池、配容器。SQLite 免运维。

**代价**：SQLite 是单写者，"两个请求同时写"要靠 WAL + 等锁超时兜住（见决策 3）；schema 变更得自己管。

**量级假设**：单机、以单进程为主，消息量万级以内。目标是"重启不丢上下文"，不追求多机共享。

### 2. `session_id` 校验

不可以直接拿它拼路径或拼 SQL。`session_id` 是调用方给的任意字符串（以后会直接从 HTTP 请求体里来），
拼路径会出现路径穿越，拼 SQL 会出现注入——这和课程第 36 课的 prompt injection 本质是同一类问题：
**外部输入不能变成代码或路径的一部分**。

结论：`session_id` 只当数据处理。SQL 全部走参数化绑定（实现见 [session_store.py](agent_lab/session_store.py)）；
将来如果真按 session 落成文件，文件名必须走白名单映射，而不是直接拼接。

### 3. 并发写

**背景**：两个请求同时写同一个 `session_id`，SQLite 默认单写者，会不会直接报 `database is locked`？

**选择**：
- 每个连接设 `timeout=10.0` + `PRAGMA busy_timeout = 10000`，让写请求**排队等待**而不是立刻失败
- 建库时切 WAL（`PRAGMA journal_mode = WAL`），读不阻塞写、写不阻塞读
- 建表 + 切 WAL 这类"一次性独占"操作，进程内用一把锁串行化

**代价**：WAL 会多出 `-wal` / `-shm` 两个伴随文件；进程内锁只覆盖同一进程，多进程首次初始化仍可能撞锁。

**怎么验证**：[test_session_store_edges.py](tests/test_session_store_edges.py) 里两个线程各写 10 条，
断言 20 条一条不少且不抛异常。

**顺带纠正两个我一开始想错的地方**：
- `sqlite3.connect()` 的 `timeout` 不是"连接存活 5 秒"，而是"拿不到锁时最多等 5 秒"，跟连接寿命无关。
- 只加 `busy_timeout` 还不够——并发首次初始化仍会报错，因为切换 journal_mode 不走 busy 等待（见踩坑记录 5）。

### 4. 历史回灌策略

不要全部塞回 prompt，只塞最近的，`20` 只是默认值。

对话越长，每轮都把全历史塞进去，token 成本线性上涨、还会挤占上下文窗口；早期无关内容过多，
反而会让模型跑偏、生成质量下降。建议的流程：

1. **滑动窗口**：只带最近几轮，系统提示词独立且始终保留
2. **按 token 预算叠加**，不是按条数：从新往旧累加，超出预算就截断
3. **窗口外的旧历史做摘要**，而不是全部丢弃
4. **优先裁剪工具输出**：体积大、时效性强，最适合折叠
5. **完整历史留在 DB 里，但不等于要放进 prompt**

### 5. 传输层用标准库 urllib，不引第三方 HTTP 库

**背景**：要给模型发 POST 请求，一般会直接上 `requests` 或 `httpx`。

**选择**：`urllib.request`。整个仓库维持"零第三方依赖"（只有 pytest 是 dev 依赖）。

**为什么**：本项目的卖点是"这条链路我自己搭过一遍"。传输层的三个关键点——超时、
重试退避、错误分类——用 `httpx` 是调三个参数，用 `urllib` 就得自己写全，而**这些正是面试会追问的地方**。
另外 CI 不需要联网装包，永远不会因为依赖源抖动而红。

**代价**：`urllib` 的 API 更啰嗦（要自己处理 `HTTPError`/`URLError`/超时三种异常），
流式响应支持也更麻烦。所以传输函数做成了可注入的 `opener`，将来真要换 `httpx`，只改一层。

### 6. 调用失败时：输入先落库，再标 `failed`

**背景**：用户发来一句话，模型调用失败（限流、超时、断网），这条消息要不要存？

**选择**：**先落库再调用**。失败时把它标成 `status=failed`，并且组 prompt 时**只带 `status=ok` 的消息**。

**为什么**：不落库的话，用户重发之前说的内容就彻底没了，事后也无法归因"到底是谁在什么时候被拒了"。
直接用 `Message` 加一个 `status` 字段承载，比另开一张"失败日志表"轻得多。

**代价**：失败的输入不会回灌给模型，所以下一轮模型看不到"用户刚说过但没被回答的那句"——
这是有意的：**没被回应的输入进上下文，会让模型以为自己已经答过了**。

**顺带**：系统提示词**不落库**，每次请求时注入。它属于配置，不属于用户对话历史，
存进去既会污染历史，也会让"按会话导出对话"变得不干净。

### 7. token 计量放独立表，按 usage 累加 + 调用前熔断

**背景**：要能回答"这个会话花了多少 token / 多少钱"，还要能设上限。

**选择**：
- 每次调用落一条 `usage_log`（session_id / model / 输入用量 / 输出用量）
- 统计**按 API 返回的 usage 累加**，不按字符数估算
- 预算熔断放在**调用之前**：超了直接拒绝，不花钱

**为什么独立表**：用量是"每次调用"的事实，不是"每条消息"的属性——一次调用既有输入用量
又有输出用量，挂在 message 上会出现两个真相来源（M0 里那个没人用的 `token_count` 字段就是这么删掉的）。

**为什么落库**：本项目的前提就是"重启后状态还在"，统计只放内存的话，重启就查不出成本了。

**代价**：多一张表和一次写入；`usage_log` 会随调用次数增长，将来要按时间归档。

**边界**：价目表是**会过期**的数据，模型不在表里时返回"费用未知"，而不是给一个错的数字。
熔断是"拒绝服务"，不是"降级回答"——**降级策略（换成便宜模型、返回兜底话术）留到后面的里程碑。**

### 8. 加字段一律走显式迁移

**背景**：M1 给 `messages` 加了 `status`。M0 时期建的库已经有数据了。

**选择**：`CREATE TABLE IF NOT EXISTS` 之后，用 `PRAGMA table_info` 检查列，
缺列就 `ALTER TABLE ... ADD COLUMN status TEXT NOT NULL DEFAULT 'ok'`。

**为什么**：`CREATE TABLE IF NOT EXISTS` 对**已存在**的表什么都不做，
所以在老库上直接跑新代码会报 `no such column`——而且往往是在生产上才第一次暴露。
默认值让老数据直接可读，不需要停机刷数据。

**怎么验证**：[tests/test_session_store_migration.py](tests/test_session_store_migration.py)
手工造一个没有 `status` 的老库，断言打开后能读、也能写新数据。

**代价**：迁移逻辑会随字段增加而变长；字段一多就该上正式的版本号迁移（`PRAGMA user_version`）了。

## 踩坑记录

> 真实排查过程比功能列表更有说服力。按「现象 / 根因 / 修复」记。

**1. `NameError: name 'sqlite3' is not defined`，7 个测试全挂**
现象：每个用例都在构造 `SessionStore` 时就炸。根因：漏了 `import sqlite3`，并且把 `connect` 写成了
`_connect`，所以连报错都被这个 `NameError` 盖住了。修复：补 import，改回 `sqlite3.connect`。
教训：**第一个报错往往不是真正的问题**，它是把后面的问题挡住了。

**2. SQL 片段拼接缺空格 → `OperationalError: near "session_id": syntax error`**
现象：读历史和删除都报语法错误。根因：SQL 是按行拼的字符串，`"SELECT ... content FROM messages "`
后面接 `"WHERE ..."` 时漏了空格，拼出来是 `contentFROM messagesWHERE`。修复：每个片段补空格。
教训：多行字符串拼接时，空格也要当成语法的一部分来检查。

**3. `Message.__init__() missing 1 required positional argument: 'session_id'`**
现象：写入时报缺参数。根因：`Message` 增加了 `session_id` 字段，但 `append_message` 里构造时没传。
修复：构造时补上。教训：数据结构加字段后，要顺着调用点全量过一遍。

**4. `created_at` 读回来是"读取时刻"而不是"写入时刻"**
现象：写入后隔几秒再读，时间戳变了。根因：`load_history` 的 SELECT 没取 `created_at`，
而 `Message` 该字段有 `default_factory`，于是**用默认值兜住了，静默给出错误数据**。
修复：SELECT 带上该列并回填；写入时也直接用 `msg.created_at`，让"返回的对象"和"落库的值"来自同一时刻。
教训：有默认值的字段最容易掩盖数据没取全的问题——它不会报错，只会悄悄给你错的答案。

**5. `PRAGMA journal_mode = WAL` 并发初始化时抛 `database is locked`**
现象：两个线程同时 `SessionStore(同一路径)`，其中一个在建库阶段抛 `database is locked`，
而此时 `busy_timeout` 已经是 10s。根因：切换 journal_mode 需要短暂独占，**这个动作不触发 busy 等待**，
所以等锁超时对它无效。修复：把建表 + 切 WAL 用进程内锁串行化。
教训：加超时不是万能的，得先确认这个操作到底会不会去"等锁"。

**6.（M1 待补）** 接真实 DeepSeek 跑通后，把只在真实网络里才会出现的问题记在这里——
限流、超时、返回体不是预期结构，都属于这一类。目前这些分支是用注入的假 opener 覆盖的
（[tests/test_llm_client.py](tests/test_llm_client.py)），还没被真实服务打过。

## 路线图

课程 46 课按能力域收敛成 8 个里程碑，保持课程先后顺序，每完成一个打一个 tag。

| 里程碑 | 覆盖课程 | 内容 | 状态 |
|---|---|---|---|
| M0 地基 | — | 会话持久化、测试骨架、决策记录 | ✅ 已完成 |
| M1 模型与会话闭环 | 01 02 03 07 | LLM client（OpenAI 兼容）、最小对话循环、token 计量与预算熔断 | ✅ 已完成 |
| M2 提示词与结构化输出 | 04 05 06 | prompt 注册表与版本、意图分类、系统/用户边界隔离 | 待开始 |
| M3 RAG 主链路 | 08 09 10 11 12 | 文档切分、自研向量检索、引用回填、低分兜底 | 待开始 |
| M4 检索进阶 | 13 14 15 16 | query 改写、rerank、hybrid 检索、索引缓存 | 待开始 |
| M5 工具与业务事实 | 17 18 19 20 21 22 23 24 | 工具契约与注册、订单/商品查询、澄清追问、错误降级、hooks 治理、MCP | 待开始 |
| M6 工作流与高风险动作 | 25 26 27 28 29 30 31 | 状态机 + checkpoint + resume（幂等）、退货/退款流程、HITL 审批 | 待开始 |
| M7 记忆、安全与交付 | 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 | 上下文构建与压缩、注入防御、trace、失败归因、成本治理、回归评测、交付 | 待开始 |

## 开发约定

- 一个里程碑 = 一组 commit + 一个 tag（`m0-foundation`、`m1-model-closure` …），HEAD 始终是完整版
- 完成后必须能跑通：`python -m pytest -q` 全绿，即视为该里程碑的验收
- 涉及取舍的改动，先在本文档或 `docs/adr/` 记一条决策，再写代码
- 不提交任何密钥（`.env` / `course.env` 已在 `.gitignore` 里）

## License

MIT，见 [LICENSE](LICENSE)。
