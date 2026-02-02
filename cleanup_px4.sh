#!/bin/bash
# cleanup_px4.sh - Kill stale PX4 SITL processes and remove lock files
# Run this before starting a new simulation session

echo "Cleaning up PX4 SITL processes..."

# Kill any running PX4 processes
if pgrep -x px4 > /dev/null; then
    pkill -9 px4
    echo "  Killed running PX4 processes"
else
    echo "  No PX4 processes found"
fi

# Remove lock files
if ls /tmp/px4_lock-* /tmp/px4-sock-* 2>/dev/null; then
    rm -f /tmp/px4_lock-* /tmp/px4-sock-*
    echo "  Removed PX4 lock files"
else
    echo "  No lock files found"
fi

echo "Cleanup complete. Ready to start a new simulation."
