# Un review de DALLE-2
Un repo con links, ideas, comentarios, y otras cosas lindas sobre DALLE-2 y parecidos.

`Disclaimer: Este review se va a quedar muy corto y muy obsoleto, muy rápido.` 

## Empezando por algún lado: Qué es DALLE-2?

### Video con mas información

[![Una introducción rápida al arte multi-modal con AI (DALL·E, CLIP, Diffusion, VQGAN)](https://img.youtube.com/vi/K2lA2WxhLnw/0.jpg)](https://www.youtube.com/watch?v=K2lA2WxhLnw)

### Algunos links para leer con mas tiempo

- 📹 [GLIDE: Generá y editá imágenes en segundos en base a lo que escribis (+ Repo)](https://www.youtube.com/watch?v=WG20CnktPbk)
- 💻 [DALLE-FLOW](https://github.com/jina-ai/dalle-flow)
- 💻 [DALL·E HuggingFace Demo](https://huggingface.co/spaces/dalle-mini/dalle-mini)
- 💻 [DALL·E Mega Training on Weights and Biases](https://wandb.ai/dalle-mini/dalle-mini/reports/DALL-E-Mega--VmlldzoxODMxMDI2)
- 💻 [A comprehensive guide to understand the multimodal AI art scene and create your own text-to-image (and other) pieces](https://multimodal.art/)
- 💻 [PyTTI](https://pytti-tools.github.io/pytti-book/intro.html)
- 💻 [MindsEye beta - ai art pilot](https://multimodal.art/mindseye)
- 💻 [Generate images from text with Latent Diffusion LAION-400M](https://huggingface.co/spaces/multimodalart/latentdiffusion)
- 💻 [(Reddit) AI-generated and manipulated content](https://www.reddit.com/r/MediaSynthesis)
- 💻 [Writing good VQGAN+CLIP prompts part one – basic prompts and style modifiers](https://www.unlimiteddreamco.xyz/2022/03/16/writing-good-prompts-part-1)
- 💻 [Artist Studies by @remi_durant](https://remidurant.com/artists/#)
- 💻 [DALL·E 2](https://openai.com/dall-e-2/)
- 💻 [Implementation of DALL-E 2, OpenAI's updated text-to-image synthesis neural network, in Pytorch](https://github.com/lucidrains/DALLE2-pytorch)
- 📹 [How does DALL-E 2 actually work?](https://www.youtube.com/watch?v=F1X4fHzF4mQ&ab_channel=AssemblyAI)
- 📹 [DALL-E 2 Inpainting / Editing Demo](https://www.youtube.com/watch?v=TFJLcy-pfTM&ab_channel=BakzT.Future)
- 💻 [“Infinite Images and the latent camera”](https://mirror.xyz/herndondryhurst.eth)

## Las charlas sobre ética, sesgos, conciencia, etc..

### ["Is DALL-E's art borrowed or stolen?"](https://www.engadget.com/dall-e-generative-ai-tracking-data-privacy-160034656.html)

Citando parte de lo que dice Daniel Jeffreis en [este articulo](https://danieljeffries.substack.com/p/the-fantastic-new-world-of-ai-art)

> Are these new tools stealing or borrowing art? 

> The short answer is simple:  No. 

> The long answer is a bit more complex but first we have to understand where the idea that DALLE or Midjourney are ripping off artists comes from in the first place.  And where they come from is a messy tangle of fears and misunderstandings.

> The first misconception is that these bots are simply copy-pastas.  In other words, are they just copying and pasting images?  Computer scientists have studied the problem extensively and developed ways to test the novelty and linguistic uniqueness of Large Language Models (LLMs) crafting new text versus just cloning what they learned.  You can check out one of the studies from DeepAI here but the big bold conclusion couldn't be more clear:

> "Neural language models do not simply memorize; instead they use productive processes that allow them to combine familiar parts in novel ways."

> To be fair, OpenAI found early versions of their model were capable of "image regurgitation" aka spitting out an exact copy of a learned image.  The models did that less than 1% of the time but they wanted to push it to 0% and they found effective ways to mitigate the problem.  After looking at the data they realized it was only capable of spitting out an image copy in a few edge cases.  First, if it had low quality, simple vector art that was easy to memorize.  Second, if it had 100s of duplicate pictures of the same stock photo, like a clock that was the same image with different time.  They fixed it by removing low quality images and duplicates, pushing image regurgitation to effectively zero.  Doesn't mean it's impossible but it's really, really unlikely, like DNA churning out your Doppelganger after trillions of iterations of people on the planet.

## Quien se queda con los derechos de las imágenes y es legal venderlas?

Si miramos solamente los últimos 3 años, la tecnología se está moviendo mucho mas rápido que la legislación disponible, y casi todas las herramientas generativas de arte, texto, código, etc. estan bajo algún tipo análisis sobre si es legal o no tener derechos sobre su output. Esto sigue lo que se discute en el tema anterior con el extra de que muchas de estas grandes (y malignas) empresas que tienen los recursos (capacidad de cómputo) usan como dataset cosas que estan disponibles de forma gratuita online, muchas veces a través de licencias como MIT o porque fueron subidas a alguna red social o sitio público. Hasta ahi no habría problema, si finalmente, cualquier persona puede "inspirarse" con el trabajo de alguien más, y generar algo completamente nuevo, sin afectar al primero. El problema es cuando, si tomamos el ejemplo de [GitHub Copilot](https://github.com/features/copilot), una empresa se aprovecha de los datos de los usuarios, crea algo "closed source" y después te cobra por su uso. Asi lo dice @ReinH en [este tweet](https://twitter.com/ReinH/status/1539626662274269185) o en [discusiones de Reddit](https://www.reddit.com/r/linux/comments/vidjop/github_copilot_legally_stealingselling_licensed/):

> github copilot is incredible. it just sells code other people wrote, but because it's an "AI" it is apparently allowed to launder that code without it being a "derivative work". lol. lmao. what an amazing grift.

y, recordemos que:

> 1. The MIT license, the most popular license on github, requires attribution
> 2. Copilot has been shown to reproduce substantial portions of its sources verbatim

Esto ha provocado que algunas organizaciones decidan directamente [dejar de usar GitHub](https://sfconservancy.org/blog/2022/jun/30/give-up-github-launch/) en respuesta.

> "after all, for its first year of existence, Copilot appeared to be more research prototype than product. Facts changed last week when GitHub announced Copilot as a commercial, for-profit product. Launching a for-profit product that disrespects the FOSS community in the way Copilot does simply makes the weight of GitHub's bad behavior too much to bear..."

Pero volviendo al tema de las imágenes, como lo dice Daniel en su [post](https://danieljeffries.substack.com/p/the-fantastic-new-world-of-ai-art)

> To start with, people calling for legal solutions to this problem seem to be missing the fact that we have a lot of precedent in law already.  Under US law, you can't copyright a style and other countries take a strikingly similar stance.  The [art's law blog](https://www.artslaw.com.au/article/i-like-your-style-part-i-copyright-infringement-or-not/) puts it like this:

> "In a recent decision of the Full Federal Court, the Court reaffirmed the fundamental legal principle that copyright does not protect ideas and concepts but only the particular form in which they are expressed.[2] The effect of this principle is that you cannot copyright a style or technique. Copyright only protects you from someone else reproducing one of your actual artworks – not from someone else coming up with their own work in the same style."

> That means a photographer or artist can't sue Time magazine when they hire a junior photographer who's learned to mimic a famous photographer's style.  Of course, magazines and advertising folks do this all the time.  They find someone who can do it well enough and hire them to take the pictures.


## Lo nuevo que está dando vuelta..
### NUWA-Infinity: Autoregressive over Autoregressive Generation for Infinite Visual Synthesis
> NUWA-Infinity is our new multimodal generative model that is able to generate high-quality images and videos from given text or image input. We can generate images with resolution up to 38912 × 2048 pixels. 
> - check demo here: https://nuwa-infinity.microsoft.com
> - abs: https://arxiv.org/abs/2207.09814

https://user-images.githubusercontent.com/78419164/183158348-48e914d6-f7eb-457e-a0b5-28582382f057.mp4

