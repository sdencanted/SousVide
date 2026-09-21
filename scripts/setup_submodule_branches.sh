#!/usr/bin/env bash
# Recreate the development branch layout on a fresh SousVide clone.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

fail() {
    printf 'Error: %s\n' "$*" >&2
    exit 1
}

# Refuse existing work that a submodule update could move or merge.
[[ -z "$(git status --porcelain --untracked-files=no)" ]] ||
    fail "Tracked changes or changed submodule commits found. Use a fresh or clean clone."
git submodule foreach --quiet --recursive '
    if test -n "$(git status --porcelain --untracked-files=no)"; then
        echo "Tracked changes found in $displaypath. Commit or stash them first." >&2
        exit 1
    fi
'

# Remember which repositories are new: Git may create a default branch at the
# remote tip while cloning, even though it checks out an older pinned commit.
new_repos="|"
for path in FiGS FiGS/acados FiGS/nerfstudio FiGS/v2e; do
    if [[ ! -e "$path/.git" ]]; then
        new_repos+="$path|"
    fi
done

# Preserve branches on repeat runs; use parent-pinned commits, not --remote.
git submodule update --init --recursive --merge

attach_branch() {
    local path="$1" branch="$2" upstream="$3"
    if ! git -C "$path" show-ref --verify --quiet "refs/remotes/$upstream"; then
        git -C "$path" fetch origin
    fi
    git -C "$path" show-ref --verify --quiet "refs/remotes/$upstream" ||
        fail "$path: upstream $upstream is unavailable."

    if git -C "$path" show-ref --verify --quiet "refs/heads/$branch"; then
        if [[ "$(git -C "$path" rev-parse "refs/heads/$branch")" != "$(git -C "$path" rev-parse HEAD)" ]]; then
            if [[ "$new_repos" == *"|$path|"* ]] ||
               { ! git -C "$path" symbolic-ref -q HEAD >/dev/null &&
                 [[ "$(git -C "$path" rev-parse "refs/heads/$branch")" == "$(git -C "$path" rev-parse "$upstream")" ]]; }; then
                # Reposition a clone default branch; its tip is retained on origin.
                git -C "$path" branch -f "$branch" HEAD
            else
                fail "$path: existing $branch points elsewhere; resolve manually. No existing branch was reset."
            fi
        fi
        git -C "$path" switch "$branch"
    else
        git -C "$path" switch -c "$branch"
    fi
    git -C "$path" branch --set-upstream-to="$upstream" "$branch"
    # FiGS's 2main must push to origin/main, not origin/2main.
    git -C "$path" config --local push.default upstream
}

attach_branch . main origin/main
attach_branch FiGS 2main origin/main
attach_branch FiGS/acados master origin/master
attach_branch FiGS/nerfstudio main origin/main
attach_branch FiGS/v2e master origin/master

# Configure each submodule in its immediate parent's local Git config.
git config --local submodule.FiGS.update merge
git -C FiGS config --local submodule.acados.update merge
git -C FiGS config --local submodule.nerfstudio.update merge
git -C FiGS config --local submodule.v2e.update merge

printf '\nDevelopment branches are ready at the pinned commits.\n'
printf 'Commit and push submodule changes before committing their pointers in the parent repository.\n'
