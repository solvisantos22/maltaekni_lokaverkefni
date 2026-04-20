# Sölvi Next Steps

## Immediate

1. Share `docs/retrieval_contract.md` with Jóhannes and ask him to return retrieval results in that shape.
2. Use `docs/mock_retrieval_result.json` while his retrieval code is still in progress.
3. Test the prompt in `docs/answer_prompt.md` manually on 2-3 questions.

## First Test Questions

- Hvað get ég gert ef vara sem ég keypti er gölluð?
- Hver er munurinn á skilarétti og rétti vegna galla?
- Hvaða upplýsingar á seljandi að gefa mér við netkaup?

## Success Criteria

The first version is good enough when it returns:

- short Icelandic answer
- clear citations like `[1]` and `[2]`
- visible source snippets
- uncertainty message when the chunks do not support the question
