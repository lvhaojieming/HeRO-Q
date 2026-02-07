MEGATRON_PATH="/path/to/Megatron-LM"
INPUT_DATA="/path/to/dataset/C4"
OUTPUT_PREFIX="C4" 
MODEL_ID="/path/to/model/" 

export PYTHONPATH=$MEGATRON_PATH:$PYTHONPATH

echo "开始处理数据..."
echo "输入文件: $INPUT_DATA"
echo "使用分词器: $MODEL_ID"


python $MEGATRON_PATH/tools/preprocess_data.py \
    --input $INPUT_DATA \
    --output-prefix $OUTPUT_PREFIX \
    --tokenizer-type HuggingFaceTokenizer \
    --tokenizer-model $MODEL_ID \
    --append-eod \
    --workers 8 \
    --log-interval 10

if [ -f "${OUTPUT_PREFIX}_text_document.bin" ]; then
    echo ""
    echo "成功！已生成以下文件："
    ls -lh ${OUTPUT_PREFIX}*
    echo ""
    echo "在训练脚本中，请设置 --data-path $(pwd)/${OUTPUT_PREFIX}_text_document"
else
    echo "处理失败，请检查报错信息。"
fi
