import os, subprocess, sys, time
srv = {"l3r1": ("tests/fixtures/tool_life_x_l3_r1_mcp_qualification_server.py", {"TOOL_LIFE_X_MCP_MODE": "ok"}),
       "tl11": ("tests/fixtures/tool_life_x_tl11_nexus_catalog_qualification_server.py", {"TOOL_LIFE_X_TL11_MODE": "ok"})}
for name, (path, env) in srv.items():
    for i in range(3):
        e = {**os.environ, **env}
        t0 = time.monotonic()
        p = subprocess.Popen([sys.executable, path], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=e)
        line = p.stdout.readline().strip()
        dt = time.monotonic() - t0
        p.kill(); p.wait()
        print(name, i, "port_line=%r" % line[:8], "%.1fs" % dt, flush=True)
