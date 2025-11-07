#!/bin/bash
git add .
git commit -m "auto update"
git pull --rebase origin main
git push

