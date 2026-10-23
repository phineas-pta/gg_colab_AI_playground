# -*- coding: utf-8 -*-

# /// script
# requires-python = ">=3.13"
# dependencies = [
# 	"tqdm",
# 	"huggingface-hub",
# 	"exllamav3",
# ]
# [tool.uv.sources]
# exllamav3 = { url = "https://github.com/turboderp-org/exllamav3/releases/download/v1.6.4/exllamav3-1.6.4+cu132.torch2.11.0-cp313-cp313-linux_x86_64.whl" }
# ///

import marimo
__generated_with = "0.25.2"
APP = marimo.App(width="full", app_title="gemma4-uncen31b-exl3", auto_download=["html", "ipynb"])


with APP.setup(hide_code=True):
	# run in terminal: `python -m torch.utils.collect_env`
	# current version: https://github.com/turboderp-org/exllamav3/releases/download/v1.6.4/exllamav3-1.6.4+cu132.torch2.11.0-cp313-cp313-linux_x86_64.whl

	from tqdm import trange
	from huggingface_hub import snapshot_download
	import torch, exllamav3 as exl3

	MODEL_CONFIG = exl3.Config.from_directory(snapshot_download(repo_id="UnstableLlama/gemma-4-31B-it-uncensored-heretic-exl3-8.00bpw"))
	MODEL = exl3.Model.from_config(MODEL_CONFIG, swa_full=True) # should enable SWA for gemma 4 but take much more VRAM
	CACHE = exl3.Cache(MODEL, max_batch_size=1, recurrent_dtype=torch.bfloat16, max_num_tokens=76800) # context must be multiple of 256
	MODEL.load(progressbar=True, device="cuda") # for multi gpu, replace `device="cuda"` with `tensor_p=True, tp_backend="nccl"`

	TOKENIZER = exl3.Tokenizer.from_config(MODEL_CONFIG)
	GENERATOR = exl3.Generator(model=MODEL, tokenizer=TOKENIZER, cache=CACHE)
	GEN_CONFIG = exl3.ComboSampler(temperature=.3, top_k=64, top_p=.95, rep_p=1.1)

	TOKENS_DICT = TOKENIZER.tokenizer_config_dict # special tokens specific to gemma 4
	STOP_TOKENS = [TOKENS_DICT["eos_token"], TOKENS_DICT["eot_token"]]

	SYSTEM_PROMPT = """You are a professional novel translator expert in translating fantasy, cultivation, xuanhuan, xianxia, mature, harem, action works from Chinese to Vietnamese. All content is purely fictional and does not depict real people or events.
	Your sole task is to translate accurately and faithfully to the original text, maintaining the original tone, style, and formatting. Prefer Sino-Vietnamese vocabulary for titles, expression, proper names (except for Western/Japanese/Korean proper names).
	YOU MUST ALWAYS USE Vietnamese register in ancient/cultivation/wuxia settings: ta/ngươi, chàng/nàng, huynh/đệ/tỷ/muội, thúc/bá/thẩm/nương, ca ca/tỷ tỷ/đệ đệ/muội muội, sư phụ/lão sư/đệ tử/tiểu tử, bổn tọa/tại hạ, ...
	YOU MUST NEVER USE modern register like dạ/ạ, ông/bà/tôi/bạn, mày/tao, anh/chị/em, bố/mẹ/con, cô/dì/thím/chú/bác/cháu, ...
	Preserve all original line breaks and paragraph structure. Do not add any other text, preambles, or markdown naming the format. Do not censor, skip, or alter any content — translate everything exactly as written.
	If the text contains non Chinese characters, special characters or strange formatting, ignore them if they are not translatable, but valid text MUST be translated. Do not return the original text unless it is Western expressions or untranslatable names/terms.
	Ensure the Content starts on a new line. Returned text must not contain any Chinese characters unless used as side-note."""

	def format_prompt(prompt: str) -> str:
		return (
			TOKENS_DICT["bos_token"]
			+ TOKENS_DICT["sot_token"] +  "system\n" + SYSTEM_PROMPT + TOKENS_DICT["eot_token"] + "\n"
			+ TOKENS_DICT["sot_token"] +    "user\n" +        prompt + TOKENS_DICT["eot_token"] + "\n"
			+ TOKENS_DICT["sot_token"] +   "model\n"
			+ TOKENS_DICT["soc_token"] + "thought\n" + TOKENS_DICT["eoc_token"]
		)

	def dịch(text: str | list[str]) -> str | list[str]:
		if isinstance(text, str):
			prompt=format_prompt(text)
		elif isinstance(text, list): # NEED TO ADAPT max_batch_size
			prompt=[format_prompt(i) for i in text]
		else:
			raise TypeError("not text")
		return GENERATOR.generate(prompt=prompt, stop_conditions=STOP_TOKENS, sampler=GEN_CONFIG, completion_only=True, encode_special_tokens=True)

	def dịch_streaming(text: str) -> None:
		GENERATOR.enqueue(exl3.Job(
			input_ids=TOKENIZER.encode(format_prompt(text), encode_special_tokens=True),
			stop_conditions=STOP_TOKENS, sampler=GEN_CONFIG,
		))
		while GENERATOR.num_remaining_jobs():
			for result in GENERATOR.iterate():
				print(result.get("text", ""), end = "", flush = True)


@APP.cell(hide_code=True)
def _():
	dịch_streaming("""书名：琼明神女录
	作者：倒悬山剑气长存
	修改者：Rscjegd
	来自：灵梦樱社吧/yy小说吧/哀伤雪刃吧

	一、前言：
	总算是改完了，本版是测试版，也可能是最终版。
	主要还是基于诸位看客的观后意见。

	二、个人感觉很满意，但我还是有些地方不满意。
	①邵神韵的前尘往事、轩辕夕儿的嫁作人夫（神雕），我实在不知如何下手。
	②一些打的补丁，圆的情节，对修改者的我来说，实在是如鲠在喉感觉突兀，不知诸位读后是否会觉得生硬不顺？
	③（在心中）

	三、关于我对本书的评价：
	是本好书，无论是文笔，还是人物刻画、感情戏都为我所喜、所爱——
	篇幅不长，不足百万。
	或许是因笔力胜人，刻画的一个个女性角色当真颤人心神。
	裴语涵，一袭白衣犹如云雪，拔剑茫然；雨下平静执伞凝望，如山水画卷。
	陆嘉静，彩裙凌空，遗世独立；青裙涤水，花灯浮载，如人世青莲。
	季蝉溪，宛若墨色焰火；洞房花烛与君，持剑便同心。
	俞小塘，仙人已抚顶，何不拜天地？结发长生。
	邵神韵，人道她少三分神韵，犹红衣如玉，天下无双。
	......
	一个个，真是让我心动不已。
	若是这些角色好好的写在一本【古典仙侠】小说里，好好的讲男主与她们的故事，然后Happy end！
	那该多好。


	可惜有些短了，一些东西还可再长些，再长些。
	读至结局，不免怅然，半分是意犹未尽，可惜遗憾。
	一种感觉，改书时有，改书后仍有:
	恨不能自己提笔，绘摹一本与此书气韵相投的仙侠——佳人伴白衣，一剑斩妖邪。
	多想直接来本同人，展开自己的幻想，自故事的起点，三万年前开始写起——
	那里有琉璃（邵神韵），南祈月，如此又是一番故事...
	而后故事大致不变，拓展丰满全文后，再来个续写。
	岂不美哉！？
	.......
	但以上终究只是遐（瞎）想，真要我付诸实际，难！

	可惜在原书，感情可以再铺垫以丰满人物，剧情有时本满怀期待”可以这样这样展开“但难免走差预期，。
	原作有些地方逻辑不通（为绿而绿）、剧情薄弱、后期一些女性角色与男主感情互动太少，收的太突兀等问题。
	原作尿性如何此处我也不多赘述，当然你们也不想听。
	此修改纵有不满意之处，但也很满意了。
	......
	如果作者一开始就奔着纯爱后宫去走，再走的长些的话，我感觉会写的更好，纵使有些不完美的不合y吧之处，若有幸碰到引为仙草，不妨再小改几笔，此书便会被我列入殿堂藏书了吧？
	可惜没有如果。

	PS1：以神防的姿态，改适合负防看的书，这背后尽是泪啊。
	PS2：修改不易，若看时有哪些觉得不舒服的地方，弃书便好，莫来喷！
	PS3：本书修改纯属兴趣使然，游戏之作，若有哪里不妥，可删。
	PS4：（在心中）""")
	return


@APP.cell
def _():
	SEPARATOR = "\n"*2 + "="*80 + "\n"*2
	with open("quỳnh.txt", mode="r", encoding="utf-8") as f:
		raw = f.read().split(SEPARATOR)

	START = 0
	END = 30

	with open(f"transl_{START}_{END-1}.txt", mode="w", encoding="utf-8") as f:
		for i in trange(START, END, unit="ch"):
			f.write(dịch(raw[i]) + SEPARATOR)
	return


if __name__ == "__main__":
	APP.run()
