-- Flatten nested log structure for YC Logging
function flatten_log(tag, timestamp, record)
    local log = record["log"]

    -- If log is a map/table with nested fields, flatten it
    if type(log) == "table" then
        -- Extract nested fields from parsed JSON
        if log["message"] then
            record["log"] = log["message"]
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
    end

    -- Clean up parser-extracted fields (avoid duplicates)
    record["level"] = nil
    record["component"] = nil
    record["logger"] = nil
    record["timestamp"] = nil
    record["message"] = nil

    return 2, timestamp, record
end
