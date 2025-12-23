-- Detect source from log content patterns
function set_source(tag, timestamp, record)
    -- Safely get log field, handle both string and non-string values
    local log = record["log"]
    if type(log) ~= "string" then
        log = ""
    end

    local source = "app"  -- default

    -- Detect based on log content patterns (use pcall for safety)
    local function safe_match(pattern)
        local ok, result = pcall(string.match, log, pattern)
        return ok and result ~= nil
    end

    if safe_match("aiogram") or safe_match("telegram") or
       safe_match("bot%.") or safe_match("dispatcher") then
        source = "bot"
    elseif safe_match("arq") or safe_match("WorkerSettings") or
           safe_match("Starting worker") then
        source = "worker"
    elseif safe_match("[Rr]edis") or safe_match("redis_version") or
           safe_match("db_keys") then
        source = "redis"
    elseif safe_match("fluent") or safe_match("inotify") then
        source = "fluent-bit"
    end

    record["source"] = source
    return 2, timestamp, record
end
