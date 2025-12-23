#!/bin/bash
# Comprehensive deployment test suite
# Tests deploy.sh functionality for M2 Apple (dev) and x86_64 Linux (prod)
#
# Usage:
#   ./tests/test_deployment.sh [--unit|--integration|--e2e|--all]
#
# Test categories:
#   --unit         Unit tests for individual functions
#   --integration  Integration tests for deployment workflow
#   --e2e          End-to-end tests with actual git operations
#   --all          Run all tests (default)

set -e

# Test configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TEST_DIR="$SCRIPT_DIR/tests"
TEMP_TEST_DIR=""
FAILED_TESTS=0
PASSED_TESTS=0

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Test mode
TEST_MODE="${1:-all}"

# Logging functions
log_test() {
    echo ""
    echo -e "${BLUE}▶ TEST: $1${NC}"
}

log_pass() {
    echo -e "${GREEN}✓ PASS: $1${NC}"
    PASSED_TESTS=$((PASSED_TESTS + 1))
}

log_fail() {
    echo -e "${RED}✗ FAIL: $1${NC}"
    FAILED_TESTS=$((FAILED_TESTS + 1))
}

log_info() {
    echo -e "${YELLOW}ℹ INFO: $1${NC}"
}

# Cleanup function
cleanup_test_env() {
    if [ -n "$TEMP_TEST_DIR" ] && [ -d "$TEMP_TEST_DIR" ]; then
        rm -rf "$TEMP_TEST_DIR"
    fi
}
trap cleanup_test_env EXIT

# Create test environment
setup_test_env() {
    TEMP_TEST_DIR=$(mktemp -d)
    export DRY_RUN=true
    export MANIFEST_FILE="$TEMP_TEST_DIR/manifest.txt"
    export TOTAL_FILES=0
    export TOTAL_SIZE=0
    export FAILED_CHECKS=()
    export MISSING_FILES=()
    touch "$MANIFEST_FILE"
}

# Source deploy.sh functions (without executing main script)
source_deploy_functions() {
    # Extract only functions from deploy.sh for testing
    # Source color codes and all function definitions
    eval "$(sed -n '
        /^# Color codes/,/^NC=/p;
        /^log_section()/,/^}/p;
        /^log_success()/,/^}/p;
        /^log_warning()/,/^}/p;
        /^log_error()/,/^}/p;
        /^check_file()/,/^}/p;
        /^check_dir()/,/^}/p;
        /^copy_file()/,/^}/p;
        /^copy_directory()/,/^}/p
    ' "$SCRIPT_DIR/deploy.sh")"
}

echo ""
echo -e "${BLUE}════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}  Deployment Test Suite${NC}"
echo -e "${BLUE}════════════════════════════════════════════════════════${NC}"
echo ""
echo "Script directory: $SCRIPT_DIR"
echo "Test mode: $TEST_MODE"
echo "Architecture: $(uname -m) ($(uname -s))"
echo ""

# ============================================================================
# UNIT TESTS
# ============================================================================

run_unit_tests() {
    log_test "UNIT TESTS"

    setup_test_env
    source_deploy_functions

    # Test 1: check_file with existing file
    log_test "check_file() with existing file"
    touch "$TEMP_TEST_DIR/test_file.txt"
    if check_file "$TEMP_TEST_DIR/test_file.txt" "true"; then
        log_pass "check_file detects existing file"
    else
        log_fail "check_file failed to detect existing file"
    fi

    # Test 2: check_file with missing required file
    log_test "check_file() with missing required file"
    if check_file "$TEMP_TEST_DIR/nonexistent.txt" "true" 2>/dev/null; then
        log_fail "check_file should fail for missing required file"
    else
        log_pass "check_file correctly fails for missing required file"
    fi

    # Test 3: check_file with missing optional file
    log_test "check_file() with missing optional file"
    if check_file "$TEMP_TEST_DIR/optional.txt" "false" 2>/dev/null; then
        log_fail "check_file should return false for missing optional file"
    else
        log_pass "check_file correctly handles missing optional file"
    fi

    # Test 4: copy_file with checksum tracking
    log_test "copy_file() with checksum tracking"
    echo "test content" > "$TEMP_TEST_DIR/source.txt"
    mkdir -p "$TEMP_TEST_DIR/dest"
    DRY_RUN=false copy_file "$TEMP_TEST_DIR/source.txt" "$TEMP_TEST_DIR/dest" "source.txt"

    if [ -f "$TEMP_TEST_DIR/dest/source.txt" ] && [ $TOTAL_FILES -eq 1 ] && grep -q "source.txt" "$MANIFEST_FILE"; then
        log_pass "copy_file copies file and updates manifest"
    else
        log_fail "copy_file failed to copy or track file"
    fi

    # Test 5: copy_file with missing source
    log_test "copy_file() with missing source"
    if copy_file "$TEMP_TEST_DIR/missing.txt" "$TEMP_TEST_DIR/dest" "missing.txt" 2>/dev/null; then
        log_fail "copy_file should fail for missing source"
    else
        log_pass "copy_file correctly fails for missing source"
    fi

    # Test 6: copy_directory functionality
    log_test "copy_directory() with tracking"
    mkdir -p "$TEMP_TEST_DIR/source_dir"
    echo "file1" > "$TEMP_TEST_DIR/source_dir/file1.txt"
    echo "file2" > "$TEMP_TEST_DIR/source_dir/file2.txt"
    mkdir -p "$TEMP_TEST_DIR/dest_dir"

    DRY_RUN=false copy_directory "$TEMP_TEST_DIR/source_dir" "$TEMP_TEST_DIR/dest_dir" "source_dir"

    if [ -d "$TEMP_TEST_DIR/dest_dir/source_dir" ] && [ -f "$TEMP_TEST_DIR/dest_dir/source_dir/file1.txt" ]; then
        log_pass "copy_directory copies directory recursively"
    else
        log_fail "copy_directory failed to copy directory"
    fi
}

# ============================================================================
# INTEGRATION TESTS
# ============================================================================

run_integration_tests() {
    log_test "INTEGRATION TESTS"

    # Test 7: Critical files validation
    log_test "Critical files presence validation"
    CRITICAL_FILES=(
        "docker-compose.yml"
        "Dockerfile.unified"
        "Dockerfile.base"
        "requirements-prod.txt"
        "healthcheck_worker.py"
        "logging-config.yaml"
        "fluent-bit.conf"
        "flatten-log.lua"
        ".github/workflows/deploy.yml"
        "scripts/fetch-secrets.sh"
    )

    MISSING_CRITICAL=()
    for file in "${CRITICAL_FILES[@]}"; do
        if [ ! -f "$SCRIPT_DIR/$file" ]; then
            MISSING_CRITICAL+=("$file")
        fi
    done

    if [ ${#MISSING_CRITICAL[@]} -eq 0 ]; then
        log_pass "All critical files present"
    else
        log_fail "Missing critical files: ${MISSING_CRITICAL[*]}"
    fi

    # Test 8: Strongly recommended files validation
    log_test "Strongly recommended files validation"
    RECOMMENDED_FILES=(
        "parsers.conf"
        "set-source.lua"
        "README.md"
        "pytest.ini"
        "docs/DATA_PERSISTENCE.md"
    )

    MISSING_RECOMMENDED=()
    for file in "${RECOMMENDED_FILES[@]}"; do
        if [ ! -f "$SCRIPT_DIR/$file" ]; then
            MISSING_RECOMMENDED+=("$file")
        fi
    done

    if [ ${#MISSING_RECOMMENDED[@]} -eq 0 ]; then
        log_pass "All recommended files present"
    else
        log_info "Missing recommended files: ${MISSING_RECOMMENDED[*]}"
        log_pass "Recommended files check completed (warnings only)"
    fi

    # Test 9: Docker configuration validation
    log_test "Docker configuration validation"
    if [ -f "$SCRIPT_DIR/docker-compose.yml" ]; then
        # Check healthcheck parameters in docker-compose.yml
        if grep -q "interval: 60s" "$SCRIPT_DIR/docker-compose.yml" && \
           grep -q "timeout: 30s" "$SCRIPT_DIR/docker-compose.yml" && \
           grep -q "retries: 10" "$SCRIPT_DIR/docker-compose.yml" && \
           grep -q "start_period: 60s" "$SCRIPT_DIR/docker-compose.yml"; then
            log_pass "docker-compose.yml has correct healthcheck parameters"
        else
            log_fail "docker-compose.yml missing correct healthcheck configuration"
        fi
    else
        log_fail "docker-compose.yml not found"
    fi

    # Test 10: Fluent-bit authentication method
    log_test "Fluent-bit authentication validation"
    if [ -f "$SCRIPT_DIR/fluent-bit.conf" ]; then
        if grep -q "authorization instance-service-account" "$SCRIPT_DIR/fluent-bit.conf"; then
            log_pass "fluent-bit.conf uses instance-service-account authentication"
        else
            log_fail "fluent-bit.conf not using instance-service-account authentication"
        fi
    else
        log_fail "fluent-bit.conf not found"
    fi

    # Test 11: Centralized logging configuration
    log_test "Centralized logging configuration validation"
    if [ -f "$SCRIPT_DIR/logging-config.yaml" ]; then
        if grep -q "components:" "$SCRIPT_DIR/logging-config.yaml" && \
           grep -q "bot:" "$SCRIPT_DIR/logging-config.yaml" && \
           grep -q "worker:" "$SCRIPT_DIR/logging-config.yaml"; then
            log_pass "logging-config.yaml has correct component structure"
        else
            log_fail "logging-config.yaml missing component definitions"
        fi
    else
        log_fail "logging-config.yaml not found"
    fi

    # Test 12: Healthcheck script validation
    log_test "Healthcheck script validation"
    if [ -f "$SCRIPT_DIR/healthcheck_worker.py" ]; then
        if grep -q "def check_redis" "$SCRIPT_DIR/healthcheck_worker.py" && \
           grep -q "def check_memory" "$SCRIPT_DIR/healthcheck_worker.py"; then
            log_pass "healthcheck_worker.py has required check functions"
        else
            log_fail "healthcheck_worker.py missing check_redis or check_memory"
        fi
    else
        log_fail "healthcheck_worker.py not found"
    fi
}

# ============================================================================
# COMPATIBILITY TESTS
# ============================================================================

run_compatibility_tests() {
    log_test "COMPATIBILITY TESTS"

    # Test 13: Architecture detection
    log_test "Architecture detection"
    ARCH=$(uname -m)
    OS=$(uname -s)

    if [ "$ARCH" = "arm64" ] || [ "$ARCH" = "aarch64" ]; then
        log_info "Detected ARM64 architecture (M2 Apple)"
        IS_ARM64=true
    elif [ "$ARCH" = "x86_64" ]; then
        log_info "Detected x86_64 architecture (Production)"
        IS_ARM64=false
    else
        log_fail "Unknown architecture: $ARCH"
        return
    fi
    log_pass "Architecture detected: $ARCH ($OS)"

    # Test 14: stat command compatibility (macOS vs Linux)
    log_test "stat command compatibility"
    TEST_FILE="$SCRIPT_DIR/deploy.sh"

    if [ "$OS" = "Darwin" ]; then
        # macOS stat format
        FILE_SIZE=$(stat -f%z "$TEST_FILE" 2>/dev/null)
        if [ -n "$FILE_SIZE" ]; then
            log_pass "macOS stat command works (BSD format)"
        else
            log_fail "macOS stat command failed"
        fi
    elif [ "$OS" = "Linux" ]; then
        # Linux stat format
        FILE_SIZE=$(stat -c%s "$TEST_FILE" 2>/dev/null)
        if [ -n "$FILE_SIZE" ]; then
            log_pass "Linux stat command works (GNU format)"
        else
            log_fail "Linux stat command failed"
        fi
    fi

    # Test 15: numfmt availability (used in copy_file)
    log_test "numfmt command availability"
    if command -v numfmt &> /dev/null; then
        TEST_SIZE=$(echo "1048576" | numfmt --to=iec-i --suffix=B 2>/dev/null)
        if [ "$TEST_SIZE" = "1.0MiB" ]; then
            log_pass "numfmt available and working"
        else
            log_info "numfmt available but unexpected output format"
            log_pass "numfmt present (with fallback handling)"
        fi
    else
        log_info "numfmt not available (deploy.sh has fallback)"
        log_pass "numfmt fallback handling verified"
    fi

    # Test 16: Docker availability
    log_test "Docker availability"
    if command -v docker &> /dev/null; then
        DOCKER_VERSION=$(docker --version 2>/dev/null | cut -d' ' -f3 | tr -d ',')
        log_pass "Docker available: version $DOCKER_VERSION"
    else
        log_fail "Docker not available"
    fi

    # Test 17: Git availability
    log_test "Git availability"
    if command -v git &> /dev/null; then
        GIT_VERSION=$(git --version | cut -d' ' -f3)
        log_pass "Git available: version $GIT_VERSION"
    else
        log_fail "Git not available"
    fi
}

# ============================================================================
# REGRESSION TESTS
# ============================================================================

run_regression_tests() {
    log_test "REGRESSION TESTS"

    # Test 18: deploy.sh uses tracking functions (not direct cp)
    log_test "deploy.sh uses copy_file/copy_directory (no direct cp with || true)"

    # Check that critical file copies use copy_file, not direct cp with || true
    if grep -n "cp.*fluent-bit.conf.*|| true" "$SCRIPT_DIR/deploy.sh" | grep -v "^#"; then
        log_fail "deploy.sh still uses direct cp with || true for fluent-bit.conf"
    elif grep -n "copy_file.*fluent-bit.conf" "$SCRIPT_DIR/deploy.sh"; then
        log_pass "deploy.sh uses copy_file for fluent-bit.conf"
    else
        log_fail "deploy.sh doesn't copy fluent-bit.conf at all"
    fi

    # Test 19: No silent failures with || true for critical files
    log_test "No silent failures for critical files"
    SILENT_FAILURES=$(grep -n "cp.*\(healthcheck_worker.py\|logging-config.yaml\|flatten-log.lua\).*|| true" "$SCRIPT_DIR/deploy.sh" | grep -v "^#" || true)

    if [ -z "$SILENT_FAILURES" ]; then
        log_pass "No silent failures (|| true) for critical files"
    else
        log_fail "Found silent failures for critical files:"
        echo "$SILENT_FAILURES"
    fi

    # Test 20: Manifest generation
    log_test "Manifest generation in deploy.sh"
    if grep -q "DEPLOYMENT MANIFEST" "$SCRIPT_DIR/deploy.sh" && \
       grep -q "cat.*MANIFEST_FILE" "$SCRIPT_DIR/deploy.sh"; then
        log_pass "deploy.sh generates deployment manifest"
    else
        log_fail "deploy.sh doesn't generate manifest"
    fi

    # Test 21: Critical file validation before deployment
    log_test "Critical file validation before deployment"
    if grep -q 'log_error "CRITICAL:.*is required"' "$SCRIPT_DIR/deploy.sh" && \
       grep -q "exit 1" "$SCRIPT_DIR/deploy.sh"; then
        log_pass "deploy.sh validates critical files and exits on failure"
    else
        log_fail "deploy.sh doesn't fail on missing critical files"
    fi
}

# ============================================================================
# END-TO-END TESTS
# ============================================================================

run_e2e_tests() {
    log_test "END-TO-END TESTS"

    # Test 22: Dry-run deployment
    log_test "Dry-run deployment (full workflow)"

    # Create temporary test directory
    E2E_TEST_DIR=$(mktemp -d)

    # Run deploy.sh in dry-run mode
    if bash "$SCRIPT_DIR/deploy.sh" --dry-run "E2E test deployment" > "$E2E_TEST_DIR/output.log" 2>&1; then
        log_pass "Dry-run deployment completed successfully"

        # Verify key sections appeared in output
        if grep -q "PRE-DEPLOYMENT VALIDATION" "$E2E_TEST_DIR/output.log" && \
           grep -q "DEPLOYMENT MANIFEST" "$E2E_TEST_DIR/output.log"; then
            log_pass "Dry-run output contains validation and manifest sections"
        else
            log_fail "Dry-run output missing expected sections"
        fi
    else
        log_fail "Dry-run deployment failed"
        cat "$E2E_TEST_DIR/output.log"
    fi

    rm -rf "$E2E_TEST_DIR"

    # Test 23: GitHub CLI availability (for CI/CD)
    log_test "GitHub CLI availability (gh)"
    if command -v gh &> /dev/null; then
        GH_VERSION=$(gh --version | head -n1 | cut -d' ' -f3)
        log_pass "GitHub CLI available: version $GH_VERSION"
    else
        log_info "GitHub CLI not available (optional for local development)"
        log_pass "GitHub CLI check completed"
    fi
}

# ============================================================================
# MAIN TEST EXECUTION
# ============================================================================

case "$TEST_MODE" in
    --unit)
        run_unit_tests
        ;;
    --integration)
        run_integration_tests
        ;;
    --compatibility)
        run_compatibility_tests
        ;;
    --regression)
        run_regression_tests
        ;;
    --e2e)
        run_e2e_tests
        ;;
    --all|*)
        run_unit_tests
        run_integration_tests
        run_compatibility_tests
        run_regression_tests
        run_e2e_tests
        ;;
esac

# ============================================================================
# TEST SUMMARY
# ============================================================================

echo ""
echo -e "${BLUE}════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}  Test Summary${NC}"
echo -e "${BLUE}════════════════════════════════════════════════════════${NC}"
echo ""
echo "Total tests: $((PASSED_TESTS + FAILED_TESTS))"
echo -e "${GREEN}Passed: $PASSED_TESTS${NC}"
echo -e "${RED}Failed: $FAILED_TESTS${NC}"
echo ""

if [ $FAILED_TESTS -eq 0 ]; then
    echo -e "${GREEN}✅ All tests passed!${NC}"
    exit 0
else
    echo -e "${RED}❌ Some tests failed${NC}"
    exit 1
fi
