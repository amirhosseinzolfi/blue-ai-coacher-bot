
(.venv) root@srv9953533997:~/blue_business# /root/blue_business/.venv/bin/python /root/blue_business/task_agent3.py
Traceback (most recent call last):
  File "/root/blue_business/task_agent3.py", line 74, in <module>
    for event in graph.stream(initial_state, {}, stream_mode="values"):
  File "/root/blue_business/.venv/lib/python3.12/site-packages/langgraph/pregel/__init__.py", line 2267, in stream
    ) = self._defaults(
        ^^^^^^^^^^^^^^^
  File "/root/blue_business/.venv/lib/python3.12/site-packages/langgraph/pregel/__init__.py", line 2079, in _defaults
    raise ValueError(
ValueError: Checkpointer requires one or more of the following 'configurable' keys: ['thread_id', 'checkpoint_ns', 'checkpoint_id']
(.venv) root@srv9953533997:~/blue_business# 