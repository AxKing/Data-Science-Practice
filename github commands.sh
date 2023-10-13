git status # Tells you what files you have in the staging area
git diff # will show you all the changes in your repository
git diff directory # will show you the changes to the files in some directory
git diff data/northern.csv

git add report.txt 
git status # checks the status of the branch
git diff -r HEAD # Compares the state of your files with those in the staging area
# -r means "Compare to a particular version"
# HEAD is a shortcut meaning "most recent commit"
# You can restrict the results to a single file or directory using
git diff -r HEAD path/to/file



nano filename #opens the filename so you can work on it.
Ctrl-K: delete a line.
Ctrl-U: un-delete a line.
Ctrl-O: save the file ('O' stands for 'output').
Ctrl-X: exit the editor.

git commit #saves changes
git commit -m "some message in quotes" # to write a commit message
#if you accidently type something you didn't want
git commit --amend - m "new message"

git log # used to view the log of the projects history
git log file/path # shows changes to a single file

git show <first few characters of the commit>
git show 0da2f7

git annotate file #shows who made the last change to each line of a file and when

git status
git add file
git commit -m "Message"

git clean -n #shows a list of files that are in the repository
git clean -f # will delete those files

git config --list # lists out configuration files
git config --list --system
git config --list --global
git config --list --local

# Changing config files
git config --global setting value
git config --global user.email rep.loop@datacamp.com


# Unstaging files that have been previously staged
git reset

# Selectively commiting changes
git add data/northern.csv # will select a single file to add to the commit
git add data/northern.csv # is like saving a file.

# How to undo changes to unstaged files.
git checkout -- filename # will discard changes that haven't been staged yet.

# How to undo changes that HAVE been staged. (NEED BOTH COMMANDS)
git reset HEAD path/to/file # unstages files
git checkout -- path/to/file # discards the changes that haven't been staged

# Restoring a file to an old version of the file
git log -3 report.txt # will give you the last 3 commits to the file
git checkout 2242bd report.txt # will reset report.txt to the commit that starts with 2242bd

cat filename # displays the updated items.

# Undoing ALL of the changes you've made
git reset directory
git reset HEAD data #will unstage any files from the data directory and revert to the previous commit.
git reset # will unstage EVERYTHING
git checkout -- data # will restore the files in the data directory to the previous state.
git checkout -- . # will revert all of the files in the current directory.

# Branches
git branch # Shows all of the branches in a repository.
git diff master..summary-statistics #will show the differences in master and summary-statistics
git checkout branch_name #switches you to another branch.
git rm file_name # removes a file


#Creating a new branch
git checkout -b new_branch_name #creates a new branch and switches to it.

# Merging branches
git merge summary-statistics master -m "merging summary statistics" #merges summary-statistics into master branch

# Fixing Merge conflicts
git merge alter-report-title master -m "empty message" #If git tells you, you have a merge conflict, run status"
git status
nano report.txt # Get rid of all the >>>>>> markers to fix the conflict.
git add report.txt
git commit -m "fixed merge conflicts"


# Creating a new repository
git init project-name #creates a new repositry called "project-name" in the current working directory.

# Turning an existing project into a repository.
git init # in the working directory
git init /path/to/file # not in the working directory

# Create a copy of an existing repository
git clone URL
git clone /existing/project #createss a clone of a directory called project
git clone /existing/project newprojectname #creates a clone of a directory and names it newprojectname

# Dude where's my repository?
git remote # will tell you where your repository came from
# Cloning a repository adds a remote called origin
git remote add remote-name URL # adds a remote
git remote rm remote-name # removes a remote
git remote add thunk /home/thunk/repo # adds a remote to home/thunk/repo named thunk

# Pulling in changes from a remote repository
git pull remote branch # gets everything in branch in the remote repository remote
git pull origin master # pulls the changes from master branch into remote repository called origin

git pull origin # tries to pull down changes from origin
git checkout -- . # git checkout -- . resets the changes you made, they are lost.
git pull # attempts to pull the repository again

# Pushing changes to a remote repository
git push remote-name branch-name
git add data/northern.csv
git commit -m "Added more northern data."
git push origin master









