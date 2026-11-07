param([Parameter(Mandatory=$true)][ValidateSet('d1','d2')][string]$Experiment)
if($Experiment -eq 'd1'){& "$PSScriptRoot\invoke_e8_finetune.ps1" -RunName m14_e8_d1_lrfix_500 -DepthWeight .10 -Resume}else{& "$PSScriptRoot\invoke_e8_finetune.ps1" -RunName m14_e8_d2_lrfix_500 -DepthWeight .15 -Resume}
