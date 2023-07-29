## FGMSA

> The entire dataset has 1247 short videos (total 1.4G) and will be made public upon acceptance

Each video has a corresponding `.json` file, which containing phrase-level alignment annotations. The structure of the `.json` file is as follows:

```json5
{
  
  "contents": {
    "sentences": [], // The sentences transcribed from the utterances in videos
    "phrase_count": [] // The count number of the phrases of each sentence
  },
  "phrases": {
    "phrase_list": [], //  list of the phrases in the above transcribed sentences  
    "TimeStamp": [] // The timestamp of each phrase in the sentences
  }
}
```