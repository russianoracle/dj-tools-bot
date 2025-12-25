#!/usr/bin/env python3
"""
Redis ARQ Queue Quality Metrics

Analyzes queue performance and job processing quality:
- Pending jobs and wait times
- Failed jobs with error details
- Expired retry jobs
- Processing time statistics

Usage:
    python scripts/queue_metrics.py
    python scripts/queue_metrics.py --detailed
"""

import asyncio
import json
import sys
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

try:
    from arq import create_pool
    from arq.connections import RedisSettings
except ImportError:
    print("❌ Error: arq not installed")
    print("   Run: pip install arq")
    sys.exit(1)


class QueueMetrics:
    """Collect and analyze ARQ queue metrics."""

    def __init__(self, redis_host: str = "localhost", redis_port: int = 6379):
        self.settings = RedisSettings(host=redis_host, port=redis_port)
        self.redis = None

    async def connect(self):
        """Connect to Redis."""
        self.redis = await create_pool(self.settings)

    async def close(self):
        """Close Redis connection."""
        if self.redis:
            await self.redis.close()

    async def get_pending_jobs(self) -> List[str]:
        """Get list of pending job IDs."""
        return await self.redis.lrange("arq:queue", 0, -1)

    async def get_job_info(self, job_id: str) -> Optional[Dict]:
        """Get job information from Redis."""
        job_key = f"arq:job:{job_id}"
        job_data = await self.redis.get(job_key)
        if job_data:
            try:
                return json.loads(job_data)
            except json.JSONDecodeError:
                return None
        return None

    async def get_result_info(self, job_id: str) -> Optional[Dict]:
        """Get job result from Redis."""
        result_key = f"arq:result:{job_id}"
        result_data = await self.redis.get(result_key)
        if result_data:
            try:
                return json.loads(result_data)
            except json.JSONDecodeError:
                return None
        return None

    async def analyze_pending_jobs(self) -> Tuple[int, List[Dict]]:
        """Analyze pending jobs and their wait times."""
        pending_job_ids = await self.get_pending_jobs()
        total_pending = len(pending_job_ids)

        job_details = []
        now = datetime.now().timestamp()

        for job_id_bytes in pending_job_ids[:50]:  # Analyze first 50 for performance
            job_id = job_id_bytes.decode() if isinstance(job_id_bytes, bytes) else job_id_bytes
            job_info = await self.get_job_info(job_id)

            if job_info:
                enqueue_time = job_info.get("enqueue_time", now)
                wait_time = now - enqueue_time

                job_details.append({
                    "job_id": job_id[:20] + "..." if len(job_id) > 20 else job_id,
                    "function": job_info.get("function", "unknown"),
                    "enqueued_at": datetime.fromtimestamp(enqueue_time).strftime("%Y-%m-%d %H:%M:%S"),
                    "wait_time_seconds": int(wait_time),
                    "wait_time_minutes": round(wait_time / 60, 1),
                })

        return total_pending, job_details

    async def analyze_failed_jobs(self) -> List[Dict]:
        """Find failed jobs by scanning results."""
        failed_jobs = []

        # Scan for result keys
        cursor = 0
        pattern = "arq:result:*"

        while True:
            cursor, keys = await self.redis.scan(cursor, match=pattern, count=100)

            for key_bytes in keys:
                key = key_bytes.decode() if isinstance(key_bytes, bytes) else key_bytes
                job_id = key.replace("arq:result:", "")

                result = await self.get_result_info(job_id)
                if result and not result.get("success", True):
                    error = result.get("result", "Unknown error")
                    failed_jobs.append({
                        "job_id": job_id[:20] + "..." if len(job_id) > 20 else job_id,
                        "function": result.get("function", "unknown"),
                        "error": str(error)[:100],  # Truncate long errors
                        "timestamp": result.get("finish_time", "unknown"),
                    })

            if cursor == 0:
                break

        return failed_jobs

    async def analyze_in_progress_jobs(self) -> Tuple[int, List[Dict]]:
        """Analyze jobs currently being processed."""
        cursor = 0
        pattern = "arq:in-progress:*"
        in_progress = []
        now = datetime.now().timestamp()

        while True:
            cursor, keys = await self.redis.scan(cursor, match=pattern, count=100)

            for key_bytes in keys:
                key = key_bytes.decode() if isinstance(key_bytes, bytes) else key_bytes
                job_id = key.replace("arq:in-progress:", "")

                job_info = await self.get_job_info(job_id)
                if job_info:
                    start_time = job_info.get("start_time", now)
                    processing_time = now - start_time

                    in_progress.append({
                        "job_id": job_id[:20] + "..." if len(job_id) > 20 else job_id,
                        "function": job_info.get("function", "unknown"),
                        "started_at": datetime.fromtimestamp(start_time).strftime("%Y-%m-%d %H:%M:%S"),
                        "processing_seconds": int(processing_time),
                        "processing_minutes": round(processing_time / 60, 1),
                    })

            if cursor == 0:
                break

        return len(in_progress), in_progress

    async def analyze_expired_retries(self) -> List[Dict]:
        """Find jobs with expired retry attempts."""
        expired = []

        # Check job keys for retry information
        cursor = 0
        pattern = "arq:job:*"
        now = datetime.now().timestamp()

        while True:
            cursor, keys = await self.redis.scan(cursor, match=pattern, count=100)

            for key_bytes in keys:
                key = key_bytes.decode() if isinstance(key_bytes, bytes) else key_bytes
                job_id = key.replace("arq:job:", "")

                job_info = await self.get_job_info(job_id)
                if job_info:
                    # Check if job has retries and is expired
                    max_tries = job_info.get("max_tries", 1)
                    current_try = job_info.get("try_count", 0)
                    defer_until = job_info.get("defer_until")

                    if defer_until and defer_until < now and current_try >= max_tries:
                        expired.append({
                            "job_id": job_id[:20] + "..." if len(job_id) > 20 else job_id,
                            "function": job_info.get("function", "unknown"),
                            "tries": f"{current_try}/{max_tries}",
                            "expired_at": datetime.fromtimestamp(defer_until).strftime("%Y-%m-%d %H:%M:%S"),
                        })

            if cursor == 0:
                break

        return expired[:50]  # Limit to 50 for display

    async def analyze_stuck_jobs(self, threshold_minutes: int = 30) -> List[Dict]:
        """
        Find stuck jobs (in-progress too long).

        Args:
            threshold_minutes: Jobs processing longer than this are considered stuck

        Returns:
            List of stuck job details
        """
        stuck = []
        threshold_seconds = threshold_minutes * 60
        now = datetime.now().timestamp()

        # Get all in-progress jobs
        _, in_progress_jobs = await self.analyze_in_progress_jobs()

        for job in in_progress_jobs:
            if job["processing_seconds"] > threshold_seconds:
                stuck.append({
                    "job_id": job["job_id"],
                    "function": job["function"],
                    "started_at": job["started_at"],
                    "processing_time": f"{job['processing_minutes']}m ({job['processing_seconds']}s)",
                    "threshold_exceeded": f"{(job['processing_seconds'] - threshold_seconds) / 60:.1f}m",
                })

        return stuck

    async def get_summary_stats(self) -> Dict:
        """Get summary statistics."""
        total_pending, _ = await self.analyze_pending_jobs()
        total_in_progress, _ = await self.analyze_in_progress_jobs()
        failed_jobs = await self.analyze_failed_jobs()
        stuck_jobs = await self.analyze_stuck_jobs(threshold_minutes=30)

        # Count completed jobs (results)
        cursor = 0
        total_completed = 0
        while True:
            cursor, keys = await self.redis.scan(cursor, match="arq:result:*", count=1000)
            total_completed += len(keys)
            if cursor == 0:
                break

        return {
            "pending": total_pending,
            "in_progress": total_in_progress,
            "completed": total_completed,
            "failed": len(failed_jobs),
            "stuck": len(stuck_jobs),
        }


def print_header(title: str):
    """Print section header."""
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}\n")


async def main(detailed: bool = False):
    """Main entry point."""
    print_header("Redis ARQ Queue Quality Metrics")

    metrics = QueueMetrics()

    try:
        await metrics.connect()
        print("✅ Connected to Redis\n")

        # Summary stats
        print("📊 Summary Statistics:")
        stats = await metrics.get_summary_stats()
        print(f"  Pending:     {stats['pending']:>6}")
        print(f"  In Progress: {stats['in_progress']:>6}")
        print(f"  Completed:   {stats['completed']:>6}")
        print(f"  Failed:      {stats['failed']:>6}")
        if stats['stuck'] > 0:
            print(f"  Stuck:       {stats['stuck']:>6} ⚠️")

        # Pending jobs analysis
        print_header("⏳ Pending Jobs & Wait Times")
        total_pending, pending_details = await metrics.analyze_pending_jobs()

        if pending_details:
            print(f"Total: {total_pending} jobs\n")
            print(f"{'Job ID':<25} {'Function':<30} {'Wait Time':<15} {'Enqueued At'}")
            print("-" * 90)

            for job in sorted(pending_details, key=lambda x: x['wait_time_seconds'], reverse=True)[:10]:
                wait = f"{job['wait_time_minutes']}m ({job['wait_time_seconds']}s)"
                print(f"{job['job_id']:<25} {job['function']:<30} {wait:<15} {job['enqueued_at']}")

            if total_pending > 10:
                print(f"\n... and {total_pending - 10} more")

            # Wait time statistics
            wait_times = [j['wait_time_seconds'] for j in pending_details]
            avg_wait = sum(wait_times) / len(wait_times) if wait_times else 0
            max_wait = max(wait_times) if wait_times else 0

            print(f"\nWait Time Stats:")
            print(f"  Average: {avg_wait / 60:.1f} minutes")
            print(f"  Maximum: {max_wait / 60:.1f} minutes")
        else:
            print("No pending jobs ✅")

        # In-progress jobs
        print_header("🔄 Jobs Currently Processing")
        total_in_progress, in_progress_details = await metrics.analyze_in_progress_jobs()

        if in_progress_details:
            print(f"Total: {total_in_progress} jobs\n")
            print(f"{'Job ID':<25} {'Function':<30} {'Processing Time':<15} {'Started At'}")
            print("-" * 90)

            for job in in_progress_details[:10]:
                proc_time = f"{job['processing_minutes']}m ({job['processing_seconds']}s)"
                print(f"{job['job_id']:<25} {job['function']:<30} {proc_time:<15} {job['started_at']}")
        else:
            print("No jobs processing ✅")

        # Stuck jobs (always show)
        print_header("⚠️  Stuck Jobs (Processing >30min)")
        stuck_jobs = await metrics.analyze_stuck_jobs(threshold_minutes=30)

        if stuck_jobs:
            print(f"Total: {len(stuck_jobs)} stuck jobs ⚠️\n")
            print(f"{'Job ID':<25} {'Function':<30} {'Processing Time':<20} {'Threshold Exceeded'}")
            print("-" * 105)

            for job in stuck_jobs[:10]:
                print(f"{job['job_id']:<25} {job['function']:<30} {job['processing_time']:<20} +{job['threshold_exceeded']}")

            if len(stuck_jobs) > 10:
                print(f"\n... and {len(stuck_jobs) - 10} more")

            print(f"\n💡 Stuck jobs may indicate:")
            print(f"   - Worker timeout issues")
            print(f"   - Deadlocks or infinite loops")
            print(f"   - Network/IO blocking")
            print(f"   - Consider killing and re-queuing")
        else:
            print("No stuck jobs ✅")

        # Failed jobs
        print_header("❌ Failed Jobs")
        failed_jobs = await metrics.analyze_failed_jobs()

        if failed_jobs:
            print(f"Total: {len(failed_jobs)} failed jobs\n")
            print(f"{'Job ID':<25} {'Function':<30} {'Error'}")
            print("-" * 90)

            for job in failed_jobs[:10]:
                print(f"{job['job_id']:<25} {job['function']:<30} {job['error']}")

            if len(failed_jobs) > 10:
                print(f"\n... and {len(failed_jobs) - 10} more")
        else:
            print("No failed jobs ✅")

        # Expired retries
        if detailed:
            print_header("⏰ Expired Retry Jobs")
            expired = await metrics.analyze_expired_retries()

            if expired:
                print(f"Total: {len(expired)} expired jobs\n")
                print(f"{'Job ID':<25} {'Function':<30} {'Tries':<10} {'Expired At'}")
                print("-" * 90)

                for job in expired:
                    print(f"{job['job_id']:<25} {job['function']:<30} {job['tries']:<10} {job['expired_at']}")
            else:
                print("No expired retry jobs ✅")

        print("\n" + "=" * 60)

    except ConnectionRefusedError:
        print("❌ Error: Cannot connect to Redis")
        print("   Make sure Redis is running:")
        print("   - Local: redis-server")
        print("   - Docker: docker exec -it $(docker ps -q -f name=redis) redis-cli ping")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        await metrics.close()


if __name__ == "__main__":
    detailed = "--detailed" in sys.argv
    asyncio.run(main(detailed=detailed))