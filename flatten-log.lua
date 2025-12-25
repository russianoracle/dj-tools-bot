-- Flatten nested log structure for YC Logging
function flatten_log(tag, timestamp, record)
    local log = record["log"]

    -- If log is empty or nil, drop the record
    if not log or (type(log) == "string" and log:match("^%s*$")) then
        return -1, timestamp, record
    end

    -- If log is a map/table with nested fields, flatten it
    if type(log) == "table" then
        -- Extract nested fields from parsed JSON
        local message_extracted = false

        if log["message"] then
            -- Ensure message is string, not table
            if type(log["message"]) == "string" then
                record["log"] = log["message"]
                message_extracted = true
            elseif type(log["message"]) == "table" then
                -- Convert table to JSON string
                local json = require("cjson")
                record["log"] = json.encode(log["message"])
                message_extracted = true
            end
        end

        if log["level"] then
            record["log_level"] = log["level"]
        end
        if log["component"] then
            record["app_component"] = log["component"]
        end
        if log["logger"] then
            record["app_logger"] = log["logger"]
        end
        if log["timestamp"] then
            record["app_timestamp"] = log["timestamp"]
        end
        if log["user_id"] then
            record["user_id"] = log["user_id"]
        end
        if log["job_id"] then
            record["job_id"] = log["job_id"]
        end
        if log["correlation_id"] then
            record["correlation_id"] = log["correlation_id"]
        end
        if log["data"] then
            record["data"] = log["data"]
        end

        -- Fallback: if message wasn't extracted, convert entire table to JSON
        if not message_extracted then
            local json = require("cjson")
            record["log"] = json.encode(log)
        end
    end

    -- Ensure log field is always a string
    if type(record["log"]) ~= "string" then
        record["log"] = "[Malformed log entry]"
    end

    -- Set default log level if missing
    if not record["log_level"] or record["log_level"] == "" then
        record["log_level"] = "INFO"
    end

    -- Clean up parser-extracted fields (avoid duplicates)
    record["level"] = nil
    record["component"] = nil
    record["logger"] = nil
    record["timestamp"] = nil
    record["message"] = nil

    return 2, timestamp, record
end
