-- Flatten nested log structure for YC Logging
function flatten_log(tag, timestamp, record)
    local log = record["log"]

    -- If log is a map/table with nested fields, flatten it
    if type(log) == "table" then
        -- Extract nested fields
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
    end

    return 2, timestamp, record
end
