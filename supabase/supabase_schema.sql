-- ================================================================
-- IA DATA SCIENTIST — Schéma Supabase (comptes & préférences)
-- ================================================================
--
-- À exécuter dans : Supabase Dashboard > SQL Editor > New query
--
-- Ce script :
--   1. Crée une table "profiles" liée aux utilisateurs Supabase Auth
--   2. Crée automatiquement un profil à chaque inscription
--   3. Protège la table avec des règles de sécurité (RLS) :
--      chaque utilisateur ne peut lire/modifier QUE son propre profil
-- ================================================================

-- ----------------------------------------------------------------
-- 1. TABLE DES PROFILS
-- ----------------------------------------------------------------

create table if not exists public.profiles (
  id uuid references auth.users(id) on delete cascade primary key,
  full_name text,
  avatar_url text,
  theme text not null default 'dark' check (theme in ('dark', 'light')),
  language text not null default 'fr' check (language in ('fr', 'en')),
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

comment on table public.profiles is
  'Préférences et informations de profil, une ligne par utilisateur Supabase Auth.';

-- ----------------------------------------------------------------
-- 2. SECURITE (ROW LEVEL SECURITY)
--
-- Sans ça, n'importe quel utilisateur connecté pourrait lire ou
-- modifier les préférences de n'importe qui d'autre.
-- ----------------------------------------------------------------

alter table public.profiles enable row level security;

drop policy if exists "Un utilisateur voit son propre profil" on public.profiles;
create policy "Un utilisateur voit son propre profil"
  on public.profiles for select
  using (auth.uid() = id);

drop policy if exists "Un utilisateur modifie son propre profil" on public.profiles;
create policy "Un utilisateur modifie son propre profil"
  on public.profiles for update
  using (auth.uid() = id);

drop policy if exists "Un utilisateur crée son propre profil" on public.profiles;
create policy "Un utilisateur crée son propre profil"
  on public.profiles for insert
  with check (auth.uid() = id);

-- ----------------------------------------------------------------
-- 3. CREATION AUTOMATIQUE DU PROFIL A L'INSCRIPTION
--
-- Dès qu'un utilisateur s'inscrit (email/mot de passe OU Google),
-- une ligne "profiles" correspondante est créée automatiquement,
-- avec son nom et sa photo si Google les a fournis.
-- ----------------------------------------------------------------

create or replace function public.handle_new_user()
returns trigger
language plpgsql
security definer set search_path = public
as $$
begin
  insert into public.profiles (id, full_name, avatar_url)
  values (
    new.id,
    coalesce(new.raw_user_meta_data ->> 'full_name', new.raw_user_meta_data ->> 'name'),
    new.raw_user_meta_data ->> 'avatar_url'
  )
  on conflict (id) do nothing;
  return new;
end;
$$;

drop trigger if exists on_auth_user_created on auth.users;
create trigger on_auth_user_created
  after insert on auth.users
  for each row execute procedure public.handle_new_user();

-- ----------------------------------------------------------------
-- 4. MISE A JOUR AUTOMATIQUE DE "updated_at"
-- ----------------------------------------------------------------

create or replace function public.handle_updated_at()
returns trigger
language plpgsql
as $$
begin
  new.updated_at = now();
  return new;
end;
$$;

drop trigger if exists on_profile_updated on public.profiles;
create trigger on_profile_updated
  before update on public.profiles
  for each row execute procedure public.handle_updated_at();
