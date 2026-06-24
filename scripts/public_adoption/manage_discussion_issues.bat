@echo off
REM © Artur Czarnecki. All rights reserved.
REM Intergrax framework – proprietary and confidential.
REM Use, modification, or distribution without written permission is prohibited.

REM Deprecated compatibility wrapper.
REM Use scripts\public_adoption\manage_curated_issues.bat instead.

call "%~dp0manage_curated_issues.bat" %*
exit /b %ERRORLEVEL%
