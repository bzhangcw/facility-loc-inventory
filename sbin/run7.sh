output=240129
fnames=(
    "data/data_0401_0inv.xlsx"
    "data/data_0401_V4_1219_0inv.xlsx"
)
method_arg="-p 1"
for ff in $fnames; do
    fn=$(basename -s .xlsx $ff)
    for i in "0" "1"; do
        cmd="python -u rounding_ray.py --fpath $ff --conf_label 7 --T 7 --pick_instance 8 --backorder $i $method_arg &> $output/$fn.7@8@${i}.log"
        echo $cmd
    done
done