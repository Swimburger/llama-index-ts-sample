import "dotenv/config";
import { VectorStoreIndex, AudioTranscriptReader } from "llamaindex";

async function main() {
  const reader = new AudioTranscriptReader();
  // Transcribe audio and store transcript in documents
  const docs = await reader.loadData({
    // You can also use a local path to an audio file, like ./sports_injuries.mp3
    audio_url:
      "https://storage.googleapis.com/aai-docs-samples/sports_injuries.mp3",
    language_code: "en_us",
  });

  // Split text and create embeddings. Store them in a VectorStoreIndex
  const index = await VectorStoreIndex.fromDocuments(docs);

  // Query the index
  const queryEngine = index.asQueryEngine();
  const response = await queryEngine.query("What is a runner's knee?");

  // Output response
  console.log(response.toString());
}

main();
