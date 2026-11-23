#!/bin/bash
# Tải các clip Mixkit vào demo_videos/ rồi kiểm tra độ dài.
# Chạy trong Terminal của bạn:  cd ~/cv-final && bash scripts/00_download_clips.sh
cd "$(dirname "$0")/.." || exit 1
mkdir -p demo_videos

while IFS='|' read -r name id; do
  [ -z "$name" ] && continue
  out="demo_videos/${name}.mp4"
  if [ -s "$out" ]; then echo "Đã có: $out"; continue; fi
  ok=0
  for q in 720 360; do
    url="https://assets.mixkit.co/videos/${id}/${id}-${q}.mp4"
    if curl -fL --retry 3 -sS -o "$out" "$url"; then ok=1; break; fi
  done
  if [ "$ok" = 1 ] && d=$(ffprobe -v error -show_entries format=duration -of csv=p=0 "$out" 2>/dev/null) && [ -n "$d" ]; then
    printf "OK   %-14s %6.1f giây\n" "$name" "$d"
  else
    rm -f "$out"
    echo "LỖI  $name (id $id) - không tải được hoặc file hỏng"
  fi
done << 'LIST_END'
rain_cars|28092
night_highway|47615
night_city|42039
snow_car|32408
day_highway|42368
freeway|1920
city_morning|29335
sunset_avenue|40635
tunnel|47629
fog_curve|48490
forest_road|45312
LIST_END

echo
echo "Các file trong demo_videos:"
ls -lh demo_videos/*.mp4 2>/dev/null
