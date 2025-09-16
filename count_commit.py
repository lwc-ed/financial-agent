import subprocess

def get_all_commit_counts():
    try:
        result = subprocess.run(
            ['git', 'shortlog', '-s', '-n', '--all'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True
        )

        output = result.stdout.strip()
        if not output:
            print("沒有找到任何 commit 記錄。")
            return

        print("所有人的 commit 次數統計如下：\n")
        for line in output.splitlines():
            parts = line.strip().split('\t')
            if len(parts) == 2:
                count, name = parts
            else:
                count = parts[0].strip().split()[0]
                name = ' '.join(parts[0].strip().split()[1:])
            print(f"{name}: {count} 次")

    except subprocess.CalledProcessError as e:
        print("執行 git 指令失敗：", e.stderr)

# 執行統計
get_all_commit_counts()