import React, { useState, useRef, useEffect } from "react";
import {
  Send,
  Sparkles,
  Database,
  Loader2,
  ChevronRight,
  BookOpen,
  Brain,
  Zap,
} from "lucide-react";
import { fetchStats, queryGraph } from "../services/api";

const GraphRAGFrontend = () => {
  const [messages, setMessages] = useState([
    {
      type: "assistant",
      content:
        "Hello! I'm your Knowledge Graph assistant. Ask me anything about people, places, or events in my database.",
      entities: [],
    },
  ]);
  const [input, setInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [stats, setStats] = useState({ triplets: 0, entities: 0 });
  const messagesEndRef = useRef(null);
  const inputRef = useRef(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  useEffect(() => {
    fetchStats().then((data) =>
      setStats({
        triplets: data.triplets || 0,
        entities: data.entities || 0,
      })
    );
  }, []);

  const handleSubmit = async () => {
    if (!input.trim() || isLoading) return;

    setMessages((prev) => [...prev, { type: "user", content: input }]);
    setIsLoading(true);
    setInput("");

    try {
      const data = await queryGraph(input);
      setMessages((prev) => [
        ...prev,
        {
          type: "assistant",
          content: data.answer || "No answer found.",
          entities: data.entities_found || [],
          numTriplets: data.num_triplets || 0,
        },
      ]);
    } catch {
      setMessages((prev) => [
        ...prev,
        {
          type: "assistant",
          isError: true,
          content: "Backend not reachable. Is FastAPI running?",
        },
      ]);
    } finally {
      setIsLoading(false);
    }
  };

  const suggestions = [
    "Who is Barack Obama?",
    "Tell me about Jackie Chan",
    "What professions exist?",
    "Who won the 52nd Grammy Award?",
  ];

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900 text-white">
      <div className="relative max-w-6xl mx-auto px-4 py-8">
        <div className="text-center mb-8">
          <div className="inline-flex items-center gap-3 px-6 py-3 bg-white/10 rounded-full">
            <Brain className="text-purple-400" />
            <h1 className="text-2xl font-bold bg-gradient-to-r from-purple-400 to-blue-400 bg-clip-text text-transparent">
              Knowledge Graph Explorer
            </h1>
            <Database className="text-blue-400" />
          </div>

          <div className="flex justify-center gap-6 mt-4 text-sm">
            <div className="flex gap-2">
              <BookOpen className="text-green-400" />
              {stats.entities} entities
            </div>
            <div className="flex gap-2">
              <Zap className="text-yellow-400" />
              {stats.triplets} facts
            </div>
          </div>
        </div>

        <div className="bg-white/5 rounded-3xl border border-white/10 overflow-hidden">
          <div className="h-[500px] overflow-y-auto p-6 space-y-4">
            {messages.map((msg, i) => (
              <div
                key={i}
                className={`flex ${
                  msg.type === "user" ? "justify-end" : "justify-start"
                }`}
              >
                <div className="max-w-3xl px-6 py-4 rounded-2xl bg-white/10">
                  <p>{msg.content}</p>

                  {msg.entities?.length > 0 && (
                    <div className="flex gap-2 mt-3 flex-wrap">
                      {msg.entities.slice(0, 5).map((e, i) => (
                        <span
                          key={i}
                          className="text-xs px-3 py-1 bg-purple-500/30 rounded-full"
                        >
                          {e}
                        </span>
                      ))}
                    </div>
                  )}
                </div>
              </div>
            ))}

            {isLoading && (
              <Loader2 className="animate-spin text-purple-400" />
            )}
            <div ref={messagesEndRef} />
          </div>

          <div className="p-6 border-t border-white/10">
            <div className="flex gap-3">
              <input
                ref={inputRef}
                value={input}
                onChange={(e) => setInput(e.target.value)}
                placeholder="Ask about the knowledge graph..."
                className="flex-1 bg-transparent border border-white/20 rounded-xl px-4 py-3"
              />
              <button
                onClick={handleSubmit}
                className="bg-gradient-to-r from-purple-600 to-blue-600 px-6 rounded-xl"
              >
                <Send />
              </button>
            </div>

            <div className="grid grid-cols-2 gap-2 mt-4">
              {suggestions.map((s, i) => (
                <button
                  key={i}
                  onClick={() => setInput(s)}
                  className="text-left text-sm px-3 py-2 bg-white/5 rounded-lg"
                >
                  <ChevronRight className="inline mr-2 text-purple-400" />
                  {s}
                </button>
              ))}
            </div>
          </div>
        </div>

        <p className="text-center text-xs text-gray-400 mt-6">
          Powered by Graph RAG • Neo4j • LLaMA 3
        </p>
      </div>
    </div>
  );
};

export default GraphRAGFrontend;
