#!/bin/bash
# Documentation Command Verification Script
# Verifies that all commands in documentation actually work

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Counters
TOTAL=0
PASSED=0
FAILED=0

# Helper functions
pass() {
    echo -e "${GREEN}✓${NC} $1"
    ((PASSED++))
    ((TOTAL++))
}

fail() {
    echo -e "${RED}✗${NC} $1"
    ((FAILED++))
    ((TOTAL++))
}

warn() {
    echo -e "${YELLOW}!${NC} $1"
}

section() {
    echo ""
    echo "========================================="
    echo "$1"
    echo "========================================="
}

# Test functions

test_environment() {
    section "Testing Environment"
    
    # Docker
    if docker --version >/dev/null 2>&1; then
        pass "Docker installed"
    else
        fail "Docker not installed"
    fi
    
    # Docker Compose
    if docker-compose --version >/dev/null 2>&1; then
        pass "Docker Compose installed"
    else
        fail "Docker Compose not installed"
    fi
    
    # WSL2 (if on Windows)
    if grep -qi microsoft /proc/version 2>/dev/null; then
        pass "Running in WSL2"
    else
        warn "Not in WSL2 (OK if on Linux)"
    fi
}

test_project_structure() {
    section "Testing Project Structure"
    
    # Check key directories
    if [ -d "core" ]; then
        pass "core/ directory exists"
    else
        fail "core/ directory missing"
    fi
    
    if [ -d "strategies" ]; then
        pass "strategies/ directory exists"
    else
        fail "strategies/ directory missing"
    fi
    
    if [ -d ".doc" ]; then
        pass ".doc/ directory exists"
    else
        fail ".doc/ directory missing"
    fi
    
    # Check key files
    if [ -f "docker-compose.yml" ]; then
        pass "docker-compose.yml exists"
    else
        fail "docker-compose.yml missing"
    fi
    
    if [ -f "Dockerfile.dev" ]; then
        pass "Dockerfile.dev exists"
    else
        fail "Dockerfile.dev missing"
    fi
}

test_documentation() {
    section "Testing Documentation"
    
    # Check core docs
    docs=("ORIGIN.md" "INDEX.md" "INSTALL.md" "HACKING.md" "ARCHITECTURE.md" "CHANGELOG.md")
    for doc in "${docs[@]}"; do
        if [ -f ".doc/$doc" ]; then
            pass ".doc/$doc exists"
        else
            fail ".doc/$doc missing"
        fi
    done
    
    # Check ADRs
    adrs=("001-docker.md" "002-wsl2.md" "003-dns.md")
    for adr in "${adrs[@]}"; do
        if [ -f ".doc/adr/$adr" ]; then
            pass ".doc/adr/$adr exists"
        else
            fail ".doc/adr/$adr missing"
        fi
    done
    
    # Check context metadata
    if [ -f ".doc/.context/index.yaml" ]; then
        pass ".doc/.context/index.yaml exists"
    else
        fail ".doc/.context/index.yaml missing"
    fi
    
    if [ -f ".doc/.context/modules.yaml" ]; then
        pass ".doc/.context/modules.yaml exists"
    else
        fail ".doc/.context/modules.yaml missing"
    fi
    
    if [ -f ".doc/.context/DESIGN.md" ]; then
        pass ".doc/.context/DESIGN.md exists"
    else
        fail ".doc/.context/DESIGN.md missing"
    fi
}

test_docker_commands() {
    section "Testing Docker Commands"
    
    # Test docker-compose config
    if docker-compose config >/dev/null 2>&1; then
        pass "docker-compose.yml is valid"
    else
        fail "docker-compose.yml has errors"
    fi
    
    # Check if container is running
    if docker ps | grep -q godzilla-dev; then
        pass "godzilla-dev container is running"
        
        # Test exec into container
        if docker-compose exec -T app echo "test" >/dev/null 2>&1; then
            pass "Can exec into container"
        else
            fail "Cannot exec into container"
        fi
        
        # Test Python in container
        if docker-compose exec -T app python3 --version >/dev/null 2>&1; then
            pass "Python available in container"
        else
            fail "Python not available in container"
        fi
        
        # Test CMake in container
        if docker-compose exec -T app cmake --version >/dev/null 2>&1; then
            pass "CMake available in container"
        else
            fail "CMake not available in container"
        fi
        
        # Test GCC in container
        if docker-compose exec -T app gcc --version >/dev/null 2>&1; then
            pass "GCC available in container"
        else
            fail "GCC not available in container"
        fi
        
    else
        warn "godzilla-dev container not running (skipping container tests)"
        warn "Run 'docker-compose up -d' to start container"
    fi
}

test_documentation_links() {
    section "Testing Documentation Links"
    
    # Check internal links in markdown files
    # This is a simple check - just verify referenced files exist
    
    local broken_links=0
    
    # Check links in README.md
    if [ -f "README.md" ]; then
        # Extract markdown links: [text](path)
        links=$(grep -oP '\[.*?\]\(\K[^)]+' README.md | grep '\.md$' || true)
        for link in $links; do
            # Remove leading ./
            link=${link#./}
            if [ -f "$link" ]; then
                pass "README.md link valid: $link"
            else
                fail "README.md broken link: $link"
                ((broken_links++))
            fi
        done
    fi
    
    # Check links in INDEX.md
    if [ -f ".doc/INDEX.md" ]; then
        links=$(grep -oP '\[.*?\]\(\K[^)]+' .doc/INDEX.md | grep '\.md$' || true)
        for link in $links; do
            link=${link#./}
            # Resolve relative to .doc/
            if [ -f ".doc/$link" ]; then
                pass "INDEX.md link valid: $link"
            else
                fail "INDEX.md broken link: $link"
                ((broken_links++))
            fi
        done
    fi
    
    if [ $broken_links -eq 0 ]; then
        pass "No broken documentation links found"
    fi
}

test_yaml_syntax() {
    section "Testing YAML Syntax"
    
    # Check if yq or python yaml parser is available
    if command -v yq >/dev/null 2>&1; then
        if yq eval .doc/.context/index.yaml >/dev/null 2>&1; then
            pass "index.yaml is valid"
        else
            fail "index.yaml has syntax errors"
        fi
        
        if yq eval .doc/.context/modules.yaml >/dev/null 2>&1; then
            pass "modules.yaml is valid"
        else
            fail "modules.yaml has syntax errors"
        fi
    elif command -v python3 >/dev/null 2>&1; then
        if python3 -c "import yaml; yaml.safe_load(open('.doc/.context/index.yaml'))" 2>/dev/null; then
            pass "index.yaml is valid (checked with Python)"
        else
            fail "index.yaml has syntax errors"
        fi
        
        if python3 -c "import yaml; yaml.safe_load(open('.doc/.context/modules.yaml'))" 2>/dev/null; then
            pass "modules.yaml is valid (checked with Python)"
        else
            fail "modules.yaml has syntax errors"
        fi
    else
        warn "No YAML parser available (install yq or python yaml)"
    fi
}

# Main execution

echo "Documentation Verification Script"
echo "=================================="
echo ""

# Change to project root
cd "$(dirname "$0")/../.."

# Run tests
test_environment
test_project_structure
test_documentation
test_docker_commands
test_documentation_links
test_yaml_syntax

# Summary
section "Summary"
echo "Total tests: $TOTAL"
echo -e "${GREEN}Passed: $PASSED${NC}"
if [ $FAILED -gt 0 ]; then
    echo -e "${RED}Failed: $FAILED${NC}"
    exit 1
else
    echo -e "${GREEN}All tests passed!${NC}"
    exit 0
fi

