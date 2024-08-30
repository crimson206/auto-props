#!/bin/bash

# 푸시할 태그를 입력받기
read -p "Enter the tag you want to push: " tag

# 입력받은 태그가 비어있는지 확인
if [ -z "$tag" ]; then
  echo "Error: No tag provided."
  exit 1
fi

# 해당 태그가 존재하는지 확인
if ! git rev-parse "$tag" >/dev/null 2>&1; then
  echo "Error: Tag '$tag' does not exist."
  exit 1
fi

# 태그를 원격 리포지토리로 푸시
git push origin "$tag"

# 푸시 결과 확인
if [ $? -eq 0 ]; then
  echo "Tag '$tag' has been pushed to the remote repository."
else
  echo "Error: Failed to push the tag '$tag'."
  exit 1
fi
