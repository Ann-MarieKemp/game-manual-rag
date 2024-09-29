import { Pinecone } from '@pinecone-database/pinecone';
import { PDFLoader} from "@langchain/community/document_loaders/fs/pdf";
import { RecursiveCharacterTextSplitter } from 'langchain/text_splitter';
import { PineconeStore } from '@langchain/pinecone';
import { OpenAIEmbeddings } from '@langchain/openai';
import { ChatOpenAI } from "@langchain/openai";
import { RunnableSequence } from "@langchain/core/runnables";
import { formatDocumentsAsString } from "langchain/util/document";
import { PromptTemplate } from "@langchain/core/prompts";
import { StringOutputParser } from "@langchain/core/output_parsers";


import dotenv from 'dotenv'
const CONFIG = dotenv.config().parsed;
console.log(CONFIG, 'is config')

const init = async () => {
  const pc = new Pinecone({
    apiKey: CONFIG.PC_KEY
  });

  const indexes = await pc.listIndexes();
  console.log(indexes, 'is indexes')
  const loader = new PDFLoader(CONFIG.PDF_PATH)
  const doc = await loader.load();

  const textSplitter = new RecursiveCharacterTextSplitter({
    chunkSize: 1000,
    chunkOverlap: 200
  })

  const chunkedDocs = await textSplitter.splitDocuments(doc)

  try {
    const embeddings = new OpenAIEmbeddings();
    const index = pc.Index(CONFIG.PC_INDEX);

    // const vectorStore = await PineconeStore.fromDocuments(doc, embeddings, {
    //   pineconeIndex: index,
    //   textKey: 'text'
    // });

    const vectorStore = await PineconeStore.fromExistingIndex(embeddings, {
      pineconeIndex: index,
      textKey: "text",
    })
    const retriever = vectorStore.asRetriever();
    console.log(process.argv[2])


    const model = new ChatOpenAI({
      temperature: 1,
      modelName: 'gpt-3.5-turbo'
    })


    const questionPrompt = PromptTemplate.fromTemplate(
      // `Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer. If the question is not related to the context, politely respond that you are tuned to only answer questions that are related to the context.
      `Use the following pieces of context to answer the question at the end. 
    
      ----------------
      CONTEXT: {context}
      ----------------
      CHAT HISTORY: {chatHistory}
      ----------------
      QUESTION: {question}
      ----------------
      Helpful Answer:`
    );

    const formatChatHistory = (
      human,
      ai,
      previousChatHistory,
    ) => {
      const newInteraction = `Human: ${human}\nAI: ${ai}`;
      if (!previousChatHistory) {
        return newInteraction;
      }
      return `${previousChatHistory}\n\n${newInteraction}`;
    };


  const chain = RunnableSequence.from([
    {
      question: (input) =>
        input.question,
      chatHistory: (input) =>
        input.chatHistory ?? "",
      context: async (input) => {
        const relevantDocs = await retriever.invoke(input.question);
        const serialized = formatDocumentsAsString(relevantDocs);
        return serialized;
      },
    },
    questionPrompt,
    model,
    new StringOutputParser(),
  ]);

  const questionOne = `${process.argv[2]}`;

const resultOne = await chain.invoke({
  question: questionOne,
});
console.log({ resultOne });

const resultTwo = await chain.invoke({
  chatHistory: formatChatHistory(resultOne, questionOne),
  question: `${process.argv[3]}`,
});

if(resultTwo) {
  console.log(resultTwo)
}
} catch(err) {
  console.error(err)
}
}


init();



