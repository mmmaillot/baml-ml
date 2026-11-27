# Four trial medians; these are not per-call latency quantiles.
def med: sort | if length%2==1 then .[(length/2|floor)] else (.[length/2-1]+.[length/2])/2 end;
def stats:
  {median:(med),min:min,max:max};
group_by([.workload,.mode]) | map(
  . as $g | .[0] as $r |
  {workload:$r.workload,mode:$r.mode,calls:$r.calls,rows:$r.rows,
   trial_count:length,
   native_ms:([.[].elapsed_ns/1000000]|stats),
   mean_us_per_call: (if $r.calls>0 then ([.[].elapsed_ns/1000/$r.calls]|stats) else null end),
   rows_per_second:(if $r.rows>0 then ([.[] | .rows*1000000000/.elapsed_ns]|stats) else null end),
   process_wall_s:([.[].wall_s]|stats),
   cpu_s:([.[]|.user_s+.sys_s]|stats),
   peak_rss_kib:([.[].peak_rss_kib]|stats),
   spans:([.[].spans]|stats),
   btel_bytes:([.[].btel_bytes]|stats),cas_bytes:([.[].cas_bytes]|stats),
   index_bytes:([.[].index_bytes]|stats),
   recording_delta_bytes:([.[].recording_total_delta]|stats),
   other_bytes:([.[].other_bytes]|stats)}
) as $groups |
{groups:[$groups[] as $g | ($groups[]|select(.workload==$g.workload and .mode=="off")) as $off |
  $g + {native_ratio_vs_off:(if $g.calls>0 then $g.native_ms.median/$off.native_ms.median else null end),
        wall_ratio_vs_off:(if $off.process_wall_s.median>0 then $g.process_wall_s.median/$off.process_wall_s.median else null end)}],
 distinct_storage_slope:[
   ["off","default","empty","span"][] as $mode |
   ($groups[]|select(.workload=="distinct100" and .mode==$mode)) as $a |
   ($groups[]|select(.workload=="distinct1000" and .mode==$mode)) as $b |
   {mode:$mode,interval_calls:900,
    btel_bytes_per_additional_call:(($b.btel_bytes.median-$a.btel_bytes.median)/900),
    cas_bytes_per_additional_call:(($b.cas_bytes.median-$a.cas_bytes.median)/900),
    index_bytes_per_additional_call:(($b.index_bytes.median-$a.index_bytes.median)/900),
    total_delta_bytes_per_additional_call:(($b.recording_delta_bytes.median-$a.recording_delta_bytes.median)/900)}
 ]}
