import { Button } from "../ui/Button";

interface NewChatButtonProps {
  onClick: () => void;
}

export function NewChatButton({ onClick }: NewChatButtonProps) {
  return (
    <Button className="w-full justify-start gap-2" onClick={onClick}>
      <span className="text-lg leading-none">+</span>
      <span>New conversation</span>
    </Button>
  );
}
