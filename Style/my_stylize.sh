set -e
# Get a carriage return into `cr`
cr=`echo $'\n.'`
cr=${cr%.}

if [ "$#" -le 0 ]; then
   echo "Usage: bash my_stylize.sh <path_to_content_image>"
   exit 1
fi

device='/gpu:0'

# Parse arguments
content_image="$1"
content_dir=$(dirname "$content_image")
content_filename=$(basename "$content_image")

styles_dir="ourStyles/*"

style_weight=(50 100 500 1000 5000)


for i in $(ls -d $styles_dir); 
do 
	echo $i;
	style_dir=$(basename "$i");
	echo $style_dir;
	for j in $(ls $i);
		do
		echo ${j};
		filename="${j%.jpg}";
		echo $filename
		for w in ${style_weight[@]};
			do
			echo "style_image_output/${style_dir}/${filename}/${w}/";

			echo "Rendering stylized image with style colors. This may take a while...";
			python neural_style.py \
			--content_img "${content_filename}" \
			--content_img_dir "${content_dir}" \
			--style_imgs "${j}" \
			--style_imgs_dir "${i}" \
			--device "${device}" \
			--style_weight "${w}" \
			--img_output_dir "style_image_output/${style_dir}/${filename}/${w}/style_colors/" \
			--verbose;

			echo "Rendering stylized image with original colors. This may take a while...";
			python neural_style.py \
			--content_img "${content_filename}" \
			--content_img_dir "${content_dir}" \
			--style_imgs "${j}" \
			--style_imgs_dir "${i}" \
			--device "${device}" \
			--original_colors \
			--style_weight "${w}" \
			--img_output_dir "style_image_output/${style_dir}/${filename}/${w}/original_colors/" \
			--verbose;
		done
	done
done


