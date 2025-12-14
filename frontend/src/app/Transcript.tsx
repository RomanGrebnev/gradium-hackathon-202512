import React, { useEffect, useRef } from "react";
import clsx from "clsx";
import { ChatMessage } from "./chatHistory";

const Transcript = ({ chatHistory }: { chatHistory: ChatMessage[] }) => {
    const scrollRef = useRef<HTMLDivElement>(null);

    // Auto-scroll to bottom when new messages arrive
    useEffect(() => {
        if (scrollRef.current) {
            scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
        }
    }, [chatHistory]);

    return (
        <div className="w-full h-full min-h-0 min-w-0">
            <div
                ref={scrollRef}
                className="w-full h-full overflow-y-auto flex flex-col gap-3 p-4 bg-black/20 rounded-lg backdrop-blur-sm"
            >
                {chatHistory.map((msg, i) => {
                    if (msg.role === 'system') {
                        return (
                            <div key={i} className="self-center text-xs text-gray-400 italic my-1 opacity-70">
                                {msg.content}
                            </div>
                        );
                    }
                    return (
                        <div
                            key={i}
                            className={clsx(
                                "flex flex-col max-w-[85%]", // Increased width slightly
                                msg.role === "assistant" ? "self-start items-start" : "self-end items-end"
                            )}
                        >
                            <div className={clsx(
                                "text-xs uppercase opacity-50 mb-1",
                                msg.role === "assistant" ? "text-left" : "text-right"
                            )}>
                                {msg.role === "assistant" ? "Patient" : "You"}
                            </div>
                            <div className={clsx(
                                "p-3 rounded-lg text-sm break-words", // Added break-words
                                msg.role === "assistant"
                                    ? "bg-white/10 text-offwhite border border-white/10"
                                    : "bg-blue/20 text-blue border border-blue/20"
                            )}>
                                {msg.content}
                            </div>
                        </div>
                    );
                })}
            </div>
        </div>
    );
};

export default Transcript;
