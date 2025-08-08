#!/usr/bin/env python3
"""
데이터베이스 위치 찾기 스크립트
"""

import os
import sys
import subprocess

print("🔍 Chroma 데이터베이스 찾기 시작...\n")

# 1. 현재 디렉토리 확인
current_dir = os.getcwd()
print(f"현재 작업 디렉토리: {current_dir}")

# 2. 모든 디렉토리 검색
print("\n📁 모든 하위 디렉토리 검색 중...")
all_dirs = []
for root, dirs, files in os.walk("."):
    for dir_name in dirs:
        full_path = os.path.join(root, dir_name)
        all_dirs.append(full_path)
        
        # Chroma 관련 디렉토리 찾기
        if any(keyword in dir_name.lower() for keyword in ["chroma", "northeastern", "unified", "db"]):
            print(f"  → 관련 디렉토리 발견: {full_path}")

# 3. 숨김 파일/폴더 확인 (Mac/Linux)
print("\n🔍 숨김 파일/폴더 확인 중...")
try:
    # ls -la 명령어로 숨김 파일 확인
    result = subprocess.run(["ls", "-la"], capture_output=True, text=True)
    lines = result.stdout.split('\n')
    for line in lines:
        if 'northeastern' in line or 'chroma' in line:
            print(f"  → {line}")
except:
    pass

# 4. Chroma 기본 저장 위치 확인
print("\n📍 Chroma 기본 저장 위치 확인...")
possible_locations = [
    os.path.expanduser("~/.chroma"),
    os.path.expanduser("~/chroma"),
    "./.chroma",
    "./chroma",
    "./northeastern_unified_db_v2",
    "./northeastern_unified_db",
    "./northeastern_chroma_db",
    "./.northeastern_unified_db_v2",  # 숨김 폴더
]

for location in possible_locations:
    if os.path.exists(location):
        print(f"  ✓ 발견: {os.path.abspath(location)}")
        try:
            contents = os.listdir(location)
            print(f"    내용: {contents[:3]}..." if len(contents) > 3 else f"    내용: {contents}")
        except:
            print("    (접근 불가)")

# 5. 파일 시스템에서 .sqlite3 파일 찾기
print("\n🗄️ SQLite 데이터베이스 파일 검색 중...")
for root, dirs, files in os.walk("."):
    for file in files:
        if file.endswith(('.sqlite3', '.db', '.sqlite')):
            full_path = os.path.join(root, file)
            file_size = os.path.getsize(full_path)
            print(f"  → SQLite 파일 발견: {full_path} (크기: {file_size/1024:.1f}KB)")

# 6. chroma.sqlite3 특별 검색
print("\n🎯 chroma.sqlite3 파일 검색...")
try:
    result = subprocess.run(["find", ".", "-name", "chroma.sqlite3"], capture_output=True, text=True)
    if result.stdout:
        print(f"  ✓ 발견된 chroma.sqlite3 파일:")
        for line in result.stdout.strip().split('\n'):
            if line:
                print(f"    {line}")
except:
    pass

# 7. 실제 Chroma 데이터베이스 생성 테스트
print("\n🧪 Chroma 데이터베이스 직접 생성 테스트...")
import chromadb
from chromadb.config import Settings

# 명시적 경로로 생성
test_path = "./test_chroma_db"
print(f"  테스트 경로: {test_path}")

client = chromadb.PersistentClient(
    path=test_path,
    settings=Settings(anonymized_telemetry=False)
)

collection = client.create_collection("test")
collection.add(
    documents=["test document"],
    metadatas=[{"test": "data"}],
    ids=["test1"]
)

if os.path.exists(test_path):
    print(f"  ✓ 테스트 DB 생성 성공!")
    test_contents = os.listdir(test_path)
    print(f"  내용: {test_contents}")
    
    # 테스트 DB 삭제
    import shutil
    shutil.rmtree(test_path)
    print("  테스트 DB 삭제됨")
else:
    print(f"  ✗ 테스트 DB 생성 실패!")

print("\n" + "="*60)
print("💡 해결 방법:")
print("1. Chroma가 다른 위치에 저장하고 있을 수 있습니다.")
print("2. 권한 문제로 폴더가 생성되지 않았을 수 있습니다.")
print("3. 위에서 발견된 경로를 사용해보세요.")