#Requires -RunAsAdministrator
# © Artur Czarnecki. All rights reserved.
<#
.SYNOPSIS
  Prepare local Windows custody directories for EVOL-X B6 signing (ECAP-P4).

.DESCRIPTION
  Creates C:\ProgramData\Intergrax\evol-b6-custody with ACLs denying the
  current interactive user write access. Does NOT generate keys or activate ECAP.

  After running, an administrator must:
  1. Create dedicated attestor/custodian Windows accounts.
  2. Grant those accounts access to the custody tree only.
  3. Run key ceremony and publish trust root (public) separately.
#>
Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$Root = "C:\ProgramData\Intergrax\evol-b6-custody"
$Subdirs = @("private", "public", "pending", "signed", "audit")

New-Item -ItemType Directory -Force -Path $Root | Out-Null
foreach ($name in $Subdirs) {
    New-Item -ItemType Directory -Force -Path (Join-Path $Root $name) | Out-Null
}

$denyUser = [System.Security.Principal.WindowsIdentity]::GetCurrent().Name
$acl = Get-Acl $Root
$denyRule = New-Object System.Security.AccessControl.FileSystemAccessRule(
    $denyUser,
    "Write,Delete,ChangePermissions,TakeOwnership",
    "ContainerInherit,ObjectInherit",
    "None",
    "Deny"
)
$acl.AddAccessRule($denyRule)
Set-Acl -Path $Root -AclObject $acl

Write-Host "Custody root: $Root"
Write-Host "Denied write to: $denyUser"
Write-Host "Next: create IntergraxEvolAttestor / IntergraxEvolCustodian users and grant them Modify on subfolders only."
